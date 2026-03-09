"""LanceDB storage operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

from mcp_code_search.config import Config
from mcp_code_search.models import IndexStatus, ProjectInfo, SearchResult

logger = logging.getLogger(__name__)

TABLE_NAME = "code_chunks"
META_FILE = "meta.json"


def _make_schema(dimension: int) -> pa.Schema:
    """Create a PyArrow schema with the given vector dimension."""
    return pa.schema(
        [
            pa.field("file_path", pa.string()),
            pa.field("project_path", pa.string()),
            pa.field("language", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("name", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("content", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
        ]
    )


class Store:
    """LanceDB storage for code chunks."""

    def __init__(self, config: Config, dimension: int = 384):
        self._config = config
        self._dimension = dimension
        self._schema = _make_schema(dimension)
        self._connections: dict[str, lancedb.DBConnection] = {}
        self._tables: dict[str, lancedb.table.Table] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_db(self, project_path: str) -> lancedb.DBConnection:
        if project_path not in self._connections:
            storage = self._config.get_project_storage_path(project_path)
            self._connections[project_path] = lancedb.connect(str(storage / "lancedb"))
        return self._connections[project_path]

    def _get_table(self, project_path: str) -> lancedb.table.Table:
        if project_path not in self._tables:
            db = self._get_db(project_path)
            try:
                self._tables[project_path] = db.open_table(TABLE_NAME)
            except Exception:
                logger.debug("Table %s not found, creating new", TABLE_NAME)
                self._tables[project_path] = db.create_table(
                    TABLE_NAME, schema=self._schema
                )
        return self._tables[project_path]

    def add_chunks(
        self,
        project_path: str,
        chunks: list[dict],
    ) -> int:
        """Add chunks to the store. Returns count added."""
        if not chunks:
            return 0

        table = self._get_table(project_path)
        table.add(chunks)
        return len(chunks)

    def delete_file_chunks(self, project_path: str, file_path: str) -> None:
        """Delete all chunks for a specific file."""
        table = self._get_table(project_path)
        table.delete(f"file_path = '{_escape(file_path)}'")

    def delete_project(self, project_path: str) -> None:
        """Delete all data for a project."""
        db = self._get_db(project_path)
        if TABLE_NAME in db.list_tables():
            db.drop_table(TABLE_NAME)
        # Clear cached references
        self._tables.pop(project_path, None)

    def get_file_hashes(self, project_path: str) -> dict[str, str]:
        """Get {file_path: file_hash} map for all indexed files in a project."""
        try:
            table = self._get_table(project_path)
            if table.count_rows() == 0:
                return {}
            df = table.to_arrow().select(["file_path", "file_hash"])
            result = {}
            for batch in df.to_batches():
                fps = batch.column("file_path").to_pylist()
                fhs = batch.column("file_hash").to_pylist()
                for fp, fh in zip(fps, fhs):
                    result[fp] = fh
            return result
        except Exception:
            logger.debug("Failed to get file hashes for %s", project_path)
            return {}

    def search_vector(
        self,
        project_path: str,
        query_vector: list[float],
        limit: int = 10,
        file_pattern: str = "",
        language: str = "",
    ) -> list[SearchResult]:
        """Vector similarity search."""
        table = self._get_table(project_path)
        if table.count_rows() == 0:
            return []

        q = table.search(query_vector).limit(limit)

        if file_pattern:
            q = q.where(f"file_path LIKE '%{_escape_like(file_pattern)}%'")
        if language:
            q = q.where(f"language = '{_escape(language)}'")

        snippet_max = self._config.search.snippet_max_lines
        results = q.to_list()
        return [
            _row_to_result(
                r, score_key="_distance", invert=True, snippet_max_lines=snippet_max
            )
            for r in results
        ]

    def search_fts(
        self,
        project_path: str,
        keyword: str,
        limit: int = 10,
        file_pattern: str = "",
    ) -> list[SearchResult]:
        """Full-text keyword search."""
        table = self._get_table(project_path)
        if table.count_rows() == 0:
            return []

        self.ensure_fts_index(project_path)

        try:
            q = table.search(keyword, query_type="fts").limit(limit)

            if file_pattern:
                q = q.where(f"file_path LIKE '%{_escape_like(file_pattern)}%'")

            snippet_max = self._config.search.snippet_max_lines
            results = q.to_list()
            return [
                _row_to_result(
                    r, score_key="_score", invert=False, snippet_max_lines=snippet_max
                )
                for r in results
            ]
        except Exception as e:
            logger.warning("FTS search failed: %s", e)
            return []

    def search_hybrid(
        self,
        project_path: str,
        query_vector: list[float],
        keyword: str,
        limit: int = 10,
        file_pattern: str = "",
        language: str = "",
    ) -> list[SearchResult]:
        """Hybrid search: combine vector + FTS results with RRF."""
        vector_results = self.search_vector(
            project_path,
            query_vector,
            limit=limit * 2,
            file_pattern=file_pattern,
            language=language,
        )
        fts_results = self.search_fts(
            project_path, keyword, limit=limit * 2, file_pattern=file_pattern
        )

        return _rrf_merge(
            vector_results, fts_results, limit, k=self._config.search.rrf_k
        )

    def ensure_fts_index(self, project_path: str) -> None:
        """Create FTS index if it doesn't exist."""
        table = self._get_table(project_path)
        indices = table.list_indices()
        has_fts = any(
            getattr(idx, "index_type", "") == "FTS" or "fts" in str(idx).lower()
            for idx in indices
        )
        if not has_fts:
            try:
                table.create_fts_index("content", replace=True)
            except Exception as e:
                logger.warning("Failed to create FTS index: %s", e)

    def get_index_status(self, project_path: str) -> IndexStatus:
        """Get index statistics for a project."""
        meta = self._read_meta(project_path)
        try:
            table = self._get_table(project_path)
            total_chunks = table.count_rows()
        except Exception:
            logger.debug("Failed to count rows for %s", project_path)
            total_chunks = 0

        return IndexStatus(
            project_path=project_path,
            total_files=meta.get("total_files", 0),
            total_chunks=total_chunks,
            last_indexed=meta.get("last_indexed"),
            embedding_model=meta.get("embedding_model", ""),
        )

    def save_meta(
        self,
        project_path: str,
        total_files: int,
        embedding_model: str,
        embedding_dimension: int = 0,
    ) -> None:
        """Save project metadata."""
        storage = self._config.get_project_storage_path(project_path)
        meta_path = storage / META_FILE
        meta = {
            "project_path": project_path,
            "total_files": total_files,
            "last_indexed": datetime.now().isoformat(),
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension or self._dimension,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    def _read_meta(self, project_path: str) -> dict:
        storage = self._config.get_project_storage_path(project_path)
        meta_path = storage / META_FILE
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def list_projects(self) -> list[ProjectInfo]:
        """List all indexed projects."""
        base = self._config.storage.base_path / "projects"
        if not base.exists():
            return []

        projects = []
        for d in base.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / META_FILE
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                last_indexed = None
                if meta.get("last_indexed"):
                    try:
                        last_indexed = datetime.fromisoformat(meta["last_indexed"])
                    except (ValueError, TypeError):
                        pass

                # Get chunk count via cached connection
                total_chunks = 0
                proj_path = meta.get("project_path", "")
                if proj_path:
                    try:
                        tbl = self._get_table(proj_path)
                        total_chunks = tbl.count_rows()
                    except Exception:
                        logger.debug("Failed to count chunks for %s", proj_path)

                projects.append(
                    ProjectInfo(
                        project_path=meta.get("project_path", ""),
                        storage_path=str(d),
                        total_chunks=total_chunks,
                        last_indexed=last_indexed,
                    )
                )
        return projects

    def find_similar(
        self,
        project_path: str,
        content: str,
        embedder,
        limit: int = 5,
        exclude_file: str = "",
    ) -> list[SearchResult]:
        """Find code similar to the given content."""
        vector = embedder.embed([content])[0]
        table = self._get_table(project_path)
        if table.count_rows() == 0:
            return []

        q = table.search(vector).limit(
            limit + 5
        )  # extra to compensate for self-matches
        if exclude_file:
            q = q.where(f"file_path != '{_escape(exclude_file)}'")

        snippet_max = self._config.search.snippet_max_lines
        results = q.to_list()
        return [
            _row_to_result(
                r, score_key="_distance", invert=True, snippet_max_lines=snippet_max
            )
            for r in results
        ][:limit]


def _escape(s: str) -> str:
    """Escape single quotes for SQL filter expressions."""
    return s.replace("'", "''")


def _escape_like(s: str) -> str:
    """Escape single quotes and LIKE wildcard characters."""
    return s.replace("'", "''").replace("%", "\\%").replace("_", "\\_")


def _row_to_result(
    row: dict,
    score_key: str = "_distance",
    invert: bool = True,
    snippet_max_lines: int = 30,
) -> SearchResult:
    """Convert a LanceDB row to a SearchResult."""
    raw_score = row.get(score_key, 0)
    if invert:
        # Convert distance to similarity: smaller distance = higher score
        score = 1.0 / (1.0 + raw_score) if raw_score >= 0 else 0.0
    else:
        score = float(raw_score)

    content = row.get("content", "")
    # Truncate snippet for display
    lines = content.split("\n")
    snippet = "\n".join(lines[:snippet_max_lines])
    if len(lines) > snippet_max_lines:
        snippet += f"\n... ({len(lines) - snippet_max_lines} more lines)"

    return SearchResult(
        file_path=row.get("file_path", ""),
        name=row.get("name", ""),
        chunk_type=row.get("chunk_type", ""),
        start_line=row.get("start_line", 0),
        end_line=row.get("end_line", 0),
        score=round(score, 4),
        snippet=snippet,
        language=row.get("language", ""),
    )


def _rrf_merge(
    vector_results: list[SearchResult],
    fts_results: list[SearchResult],
    limit: int,
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion to merge two result lists."""
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(vector_results):
        key = f"{r.file_path}:{r.start_line}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        result_map[key] = r

    for rank, r in enumerate(fts_results):
        key = f"{r.file_path}:{r.start_line}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in result_map:
            result_map[key] = r

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]

    merged = []
    for key in sorted_keys:
        result = result_map[key]
        result.score = round(scores[key], 4)
        merged.append(result)

    return merged
