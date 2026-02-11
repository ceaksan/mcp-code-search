"""Indexing pipeline orchestrator."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pathspec

from mcp_code_search.chunker import chunk_file
from mcp_code_search.config import Config
from mcp_code_search.embedder import Embedder, create_embedder
from mcp_code_search.store import Store
from mcp_code_search.watcher import compute_file_hash, get_changed_files

logger = logging.getLogger(__name__)

# Binary/non-text extensions to skip
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".flac",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".o",
    ".a",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".DS_Store",
    ".gitkeep",
}


class Indexer:
    """Orchestrates the indexing pipeline."""

    def __init__(self, config: Config | None = None):
        self._config = config or Config.load()
        self._embedder: Embedder | None = None
        self._store: Store | None = None

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(self._config, dimension=self.embedder.dimension)
        return self._store

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(self._config.embedding)
        return self._embedder

    def index_directory(
        self,
        path: str,
        incremental: bool = True,
    ) -> dict:
        """Index a directory. Returns stats dict."""
        project_path = str(Path(path).resolve())
        logger.info("Indexing %s (incremental=%s)", project_path, incremental)

        # Check for dimension mismatch with stored metadata
        if incremental:
            meta = self.store._read_meta(project_path)
            stored_dim = meta.get("embedding_dimension", 0)
            if stored_dim and stored_dim != self.embedder.dimension:
                logger.warning(
                    "Embedding dimension changed (%d -> %d), forcing full reindex",
                    stored_dim,
                    self.embedder.dimension,
                )
                incremental = False

        # Scan files
        all_files = self._scan_files(project_path)
        logger.info("Found %d files to consider", len(all_files))

        if incremental:
            stored_hashes = self.store.get_file_hashes(project_path)
            changed, deleted = get_changed_files(all_files, stored_hashes)

            # Delete chunks for removed files
            for fp in deleted:
                self.store.delete_file_chunks(project_path, fp)
            logger.info(
                "%d changed, %d deleted, %d unchanged",
                len(changed),
                len(deleted),
                len(all_files) - len(changed),
            )
            files_to_index = changed
        else:
            # Full reindex
            self.store.delete_project(project_path)
            files_to_index = all_files

        if not files_to_index:
            self.store.save_meta(
                project_path, len(all_files), self._config.embedding.model
            )
            return {
                "project_path": project_path,
                "total_files": len(all_files),
                "indexed_files": 0,
                "total_chunks": 0,
                "status": "no changes",
            }

        # Process files in batches
        total_chunks = 0
        batch_size = self._config.indexing.batch_size
        all_chunk_data = []

        for fp in files_to_index:
            try:
                content = self._read_file(fp)
                if not content or not content.strip():
                    continue

                # Delete old chunks for this file if incremental
                if incremental:
                    self.store.delete_file_chunks(project_path, fp)

                file_hash = compute_file_hash(fp)
                chunks = chunk_file(
                    fp,
                    content,
                    self._config.indexing.chunk_overlap_lines,
                    self._config.indexing.max_chunk_lines,
                )

                for chunk in chunks:
                    all_chunk_data.append(
                        {
                            "file_path": fp,
                            "project_path": project_path,
                            "language": chunk.language,
                            "chunk_type": chunk.chunk_type,
                            "name": chunk.name,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "content": chunk.content,
                            "file_hash": file_hash,
                        }
                    )

            except Exception as e:
                logger.warning("Failed to process %s: %s", fp, e)

        # Embed and store in batches
        for i in range(0, len(all_chunk_data), batch_size):
            batch = all_chunk_data[i : i + batch_size]
            texts = [c["content"] for c in batch]
            vectors = self.embedder.embed(texts)

            for chunk_data, vector in zip(batch, vectors):
                chunk_data["vector"] = vector

            self.store.add_chunks(project_path, batch)
            total_chunks += len(batch)

            if (i + batch_size) % 100 == 0:
                logger.info("Indexed %d/%d chunks", i + batch_size, len(all_chunk_data))

        self.store.save_meta(project_path, len(all_files), self._config.embedding.model)

        # Rebuild FTS index after adding new data
        try:
            self.store.ensure_fts_index(project_path)
        except Exception as e:
            logger.warning("Failed to rebuild FTS index for %s: %s", project_path, e)

        return {
            "project_path": project_path,
            "total_files": len(all_files),
            "indexed_files": len(files_to_index),
            "total_chunks": total_chunks,
            "status": "completed",
        }

    def _scan_files(self, project_path: str) -> list[str]:
        """Scan directory for indexable files, respecting ignore patterns."""
        root = Path(project_path)
        ignore_patterns = self._config.indexing.ignore_patterns
        max_size = self._config.indexing.max_file_size_kb * 1024

        # Load .gitignore if exists
        gitignore_spec = None
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists():
            try:
                lines = gitignore_path.read_text().splitlines()
                gitignore_spec = pathspec.PathSpec.from_lines("gitignore", lines)
            except Exception:
                pass

        # Build pathspec from config ignore patterns
        config_spec = pathspec.PathSpec.from_lines("gitignore", ignore_patterns)

        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)

            # Filter directories in-place to skip ignored dirs
            dirnames[:] = [
                d
                for d in dirnames
                if not _should_ignore_dir(d, rel_dir, config_spec, gitignore_spec)
            ]

            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root)

                # Skip binary files
                ext = os.path.splitext(fname)[1].lower()
                if ext in BINARY_EXTENSIONS:
                    continue

                # Skip files matching ignore patterns
                if config_spec.match_file(rel_path):
                    continue
                if gitignore_spec and gitignore_spec.match_file(rel_path):
                    continue

                # Skip files exceeding size limit
                try:
                    if os.path.getsize(full_path) > max_size:
                        continue
                except OSError:
                    continue

                files.append(full_path)

        return files

    def _read_file(self, file_path: str) -> str | None:
        """Read a file, returning None if it can't be read as text."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None


def _should_ignore_dir(
    dirname: str,
    parent_rel: str,
    config_spec: pathspec.PathSpec,
    gitignore_spec: pathspec.PathSpec | None,
) -> bool:
    """Check if a directory should be skipped."""
    if dirname.startswith(".") and dirname != ".":
        # Skip hidden dirs (except .)
        if dirname not in (".github", ".vscode", ".config"):
            return True

    rel_path = os.path.join(parent_rel, dirname) if parent_rel != "." else dirname

    if config_spec.match_file(rel_path + "/"):
        return True
    if config_spec.match_file(dirname):
        return True
    if gitignore_spec:
        if gitignore_spec.match_file(rel_path + "/"):
            return True
        if gitignore_spec.match_file(dirname + "/"):
            return True

    return False
