"""MCP server with FastMCP tool definitions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_code_search.config import Config
from mcp_code_search.indexer import Indexer

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "code-search",
    instructions="Local semantic code search - AST-aware indexing with hybrid search",
)

_indexer: Indexer | None = None


def _get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        config = Config.load()
        _indexer = Indexer(config)
    return _indexer


@mcp.tool()
def search_code(
    query: str,
    limit: int = 10,
    file_pattern: str = "",
    language: str = "",
) -> str:
    """Hybrid semantic + keyword code search. Best for natural language queries like 'authentication middleware' or 'database connection pool'.

    Args:
        query: Search query (natural language or code-like)
        limit: Max results to return (default 10)
        file_pattern: Filter by file path pattern (e.g. 'auth', 'models.py')
        language: Filter by language (e.g. 'python', 'typescript')
    """
    indexer = _get_indexer()
    projects = indexer.store.list_projects()
    if not projects:
        return "No indexed projects. Use index_directory first."

    all_results = []
    query_vector = indexer.embedder.embed([query])[0]

    for proj in projects:
        results = indexer.store.search_hybrid(
            proj.project_path,
            query_vector,
            query,
            limit=limit,
            file_pattern=file_pattern,
            language=language,
        )
        all_results.extend(results)

    all_results.sort(key=lambda r: r.score, reverse=True)
    all_results = all_results[:limit]

    if not all_results:
        return f"No results found for: {query}"

    return _format_results(all_results)


@mcp.tool()
def search_text(
    keyword: str,
    limit: int = 10,
    file_pattern: str = "",
) -> str:
    """Full-text keyword search (grep alternative). Best for exact matches like function names, error messages, or specific strings.

    Args:
        keyword: Keyword to search for (exact match)
        limit: Max results to return (default 10)
        file_pattern: Filter by file path pattern
    """
    indexer = _get_indexer()
    projects = indexer.store.list_projects()
    if not projects:
        return "No indexed projects. Use index_directory first."

    all_results = []
    for proj in projects:
        results = indexer.store.search_fts(
            proj.project_path,
            keyword,
            limit=limit,
            file_pattern=file_pattern,
        )
        all_results.extend(results)

    all_results.sort(key=lambda r: r.score, reverse=True)
    all_results = all_results[:limit]

    if not all_results:
        return f"No results found for: {keyword}"

    return _format_results(all_results)


@mcp.tool()
def index_directory(
    path: str,
    incremental: bool = True,
) -> str:
    """Index a codebase directory for searching. Uses Tree-sitter AST parsing for semantic chunking.

    Args:
        path: Absolute path to the directory to index
        incremental: If True, only re-index changed files (default True)
    """
    resolved = str(Path(path).resolve())
    if not Path(resolved).is_dir():
        return f"Error: {resolved} is not a directory"

    indexer = _get_indexer()

    # Reload config with project-specific overrides
    config = Config.load(Path(resolved))
    indexer._config = config

    result = indexer.index_directory(resolved, incremental=incremental)

    return (
        f"Indexing complete:\n"
        f"  Project: {result['project_path']}\n"
        f"  Total files: {result['total_files']}\n"
        f"  Indexed files: {result['indexed_files']}\n"
        f"  Total chunks: {result['total_chunks']}\n"
        f"  Status: {result['status']}"
    )


@mcp.tool()
def get_index_status() -> str:
    """Get index statistics for all indexed projects."""
    indexer = _get_indexer()
    projects = indexer.store.list_projects()

    if not projects:
        return "No indexed projects."

    lines = ["Indexed projects:\n"]
    for proj in projects:
        status = indexer.store.get_index_status(proj.project_path)
        lines.append(f"  {status.project_path}")
        lines.append(f"    Files: {status.total_files}")
        lines.append(f"    Chunks: {status.total_chunks}")
        lines.append(f"    Model: {status.embedding_model}")
        if status.last_indexed:
            lines.append(f"    Last indexed: {status.last_indexed}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def find_similar_code(
    file_path: str,
    function_name: str = "",
    limit: int = 5,
) -> str:
    """Find code similar to a given function or file. Useful for finding duplicates or related implementations.

    Args:
        file_path: Path to the source file
        function_name: Specific function/class name to find similar code for (optional)
        limit: Max results (default 5)
    """
    resolved = str(Path(file_path).resolve())
    if not Path(resolved).is_file():
        return f"Error: {resolved} is not a file"

    indexer = _get_indexer()
    projects = indexer.store.list_projects()
    if not projects:
        return "No indexed projects. Use index_directory first."

    # Read the target file/function
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    if function_name:
        # Try to find the specific function in the file
        from mcp_code_search.chunker import chunk_file

        chunks = chunk_file(resolved, content)
        target_chunk = None
        for chunk in chunks:
            if chunk.name == function_name:
                target_chunk = chunk
                break
        if target_chunk:
            content = target_chunk.content
        else:
            return f"Function '{function_name}' not found in {resolved}"

    all_results = []
    for proj in projects:
        results = indexer.store.find_similar(
            proj.project_path,
            content,
            indexer.embedder,
            limit=limit,
            exclude_file=resolved,
        )
        all_results.extend(results)

    all_results.sort(key=lambda r: r.score, reverse=True)
    all_results = all_results[:limit]

    if not all_results:
        return "No similar code found."

    return _format_results(all_results)


@mcp.tool()
def list_projects() -> str:
    """List all indexed projects with their stats."""
    indexer = _get_indexer()
    projects = indexer.store.list_projects()

    if not projects:
        return "No indexed projects."

    lines = ["Indexed projects:\n"]
    for proj in projects:
        lines.append(f"  {proj.project_path}")
        lines.append(f"    Chunks: {proj.total_chunks}")
        if proj.last_indexed:
            lines.append(f"    Last indexed: {proj.last_indexed}")
        lines.append("")

    return "\n".join(lines)


def _format_results(results: list) -> str:
    """Format search results for display."""
    lines = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r.file_path}:{r.start_line}-{r.end_line}"
        if r.name:
            header += f" ({r.chunk_type}: {r.name})"
        if r.language:
            header += f" [{r.language}]"
        header += f" score={r.score}"

        lines.append(header)
        lines.append(r.snippet)
        lines.append("")

    return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    import argparse

    parser = argparse.ArgumentParser(description="Code search MCP server")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("mcp_code_search").setLevel(logging.DEBUG)

    mcp.run()


if __name__ == "__main__":
    main()
