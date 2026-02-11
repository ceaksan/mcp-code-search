"""Tests for the indexer module."""

import os
import tempfile
from pathlib import Path

from mcp_code_search.config import Config
from mcp_code_search.indexer import Indexer


def _make_indexer(storage_tmp: str) -> Indexer:
    config = Config()
    config.storage.base_path = Path(storage_tmp)
    config.embedding.model = "all-MiniLM-L6-v2"
    return Indexer(config)


def test_index_small_directory():
    with tempfile.TemporaryDirectory() as storage_tmp:
        with tempfile.TemporaryDirectory() as project_tmp:
            # Create test files
            Path(project_tmp, "hello.py").write_text(
                "def hello():\n    return 'world'\n"
            )
            Path(project_tmp, "util.py").write_text(
                "def add(a, b):\n    return a + b\n"
            )

            indexer = _make_indexer(storage_tmp)
            result = indexer.index_directory(project_tmp, incremental=False)

            assert result["status"] == "completed"
            assert result["total_files"] == 2
            assert result["total_chunks"] > 0


def test_incremental_indexing():
    with tempfile.TemporaryDirectory() as storage_tmp:
        with tempfile.TemporaryDirectory() as project_tmp:
            Path(project_tmp, "a.py").write_text("def a(): pass\n")

            indexer = _make_indexer(storage_tmp)
            r1 = indexer.index_directory(project_tmp, incremental=False)
            assert r1["indexed_files"] == 1

            # Without changes, no files should be re-indexed
            r2 = indexer.index_directory(project_tmp, incremental=True)
            assert r2["indexed_files"] == 0
            assert r2["status"] == "no changes"


def test_respects_gitignore():
    with tempfile.TemporaryDirectory() as storage_tmp:
        with tempfile.TemporaryDirectory() as project_tmp:
            Path(project_tmp, ".gitignore").write_text("ignored/\n")
            os.makedirs(Path(project_tmp, "ignored"))
            Path(project_tmp, "ignored", "skip.py").write_text("x = 1\n")
            Path(project_tmp, "keep.py").write_text("y = 2\n")

            indexer = _make_indexer(storage_tmp)
            result = indexer.index_directory(project_tmp, incremental=False)

            # .gitignore itself is counted as a file, but ignored/skip.py is not
            assert result["total_files"] == 2  # keep.py + .gitignore


def test_skips_binary_files():
    with tempfile.TemporaryDirectory() as storage_tmp:
        with tempfile.TemporaryDirectory() as project_tmp:
            Path(project_tmp, "code.py").write_text("x = 1\n")
            Path(project_tmp, "image.png").write_bytes(b"\x89PNG\r\n")

            indexer = _make_indexer(storage_tmp)
            result = indexer.index_directory(project_tmp, incremental=False)

            assert result["total_files"] == 1


def test_empty_directory():
    with tempfile.TemporaryDirectory() as storage_tmp:
        with tempfile.TemporaryDirectory() as project_tmp:
            indexer = _make_indexer(storage_tmp)
            result = indexer.index_directory(project_tmp, incremental=False)

            assert result["total_files"] == 0
            assert result["indexed_files"] == 0
            assert result["status"] == "no changes"


def test_large_file_skip():
    with tempfile.TemporaryDirectory() as storage_tmp:
        with tempfile.TemporaryDirectory() as project_tmp:
            # Default max is 500KB
            Path(project_tmp, "small.py").write_text("x = 1\n")
            Path(project_tmp, "large.py").write_text("x = 1\n" * 200000)  # ~1.2MB

            indexer = _make_indexer(storage_tmp)
            result = indexer.index_directory(project_tmp, incremental=False)

            assert result["total_files"] == 1


def test_configurable_batch_size():
    with tempfile.TemporaryDirectory() as storage_tmp:
        with tempfile.TemporaryDirectory() as project_tmp:
            Path(project_tmp, "a.py").write_text("def a(): pass\n")

            config = Config()
            config.storage.base_path = Path(storage_tmp)
            config.embedding.model = "all-MiniLM-L6-v2"
            config.indexing.batch_size = 1

            indexer = Indexer(config)
            result = indexer.index_directory(project_tmp, incremental=False)

            assert result["status"] == "completed"
            assert result["total_chunks"] > 0
