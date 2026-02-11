"""Tests for the store module."""

import tempfile
from pathlib import Path

from mcp_code_search.config import Config
from mcp_code_search.store import (
    TABLE_NAME,
    Store,
    _escape,
    _escape_like,
    _row_to_result,
)


DIM = 768


def _make_store(tmp_path: str, dimension: int = DIM) -> Store:
    config = Config()
    config.storage.base_path = Path(tmp_path)
    return Store(config, dimension=dimension)


def _make_chunk(file_path: str = "/test/file.py", name: str = "test_func"):
    return {
        "file_path": file_path,
        "project_path": "/test",
        "language": "python",
        "chunk_type": "function",
        "name": name,
        "start_line": 1,
        "end_line": 5,
        "content": "def test_func():\n    return 42",
        "file_hash": "abc123",
        "vector": [0.1] * DIM,
    }


def test_add_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        chunk = _make_chunk()
        store.add_chunks("/test", [chunk])

        results = store.search_vector("/test", [0.1] * DIM, limit=1)
        assert len(results) == 1
        assert results[0].name == "test_func"


def test_delete_file_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.add_chunks("/test", [_make_chunk("/test/a.py", "func_a")])
        store.add_chunks("/test", [_make_chunk("/test/b.py", "func_b")])

        store.delete_file_chunks("/test", "/test/a.py")

        results = store.search_vector("/test", [0.1] * DIM, limit=10)
        file_paths = [r.file_path for r in results]
        assert "/test/a.py" not in file_paths
        assert "/test/b.py" in file_paths


def test_get_file_hashes():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.add_chunks("/test", [_make_chunk()])

        hashes = store.get_file_hashes("/test")
        assert "/test/file.py" in hashes
        assert hashes["/test/file.py"] == "abc123"


def test_fts_search():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.add_chunks("/test", [_make_chunk()])

        results = store.search_fts("/test", "test_func", limit=1)
        assert len(results) >= 1


def test_empty_search():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        results = store.search_vector("/test", [0.1] * DIM, limit=1)
        assert len(results) == 0


def test_escape_like():
    assert _escape_like("hello") == "hello"
    assert _escape_like("100%") == "100\\%"
    assert _escape_like("file_name") == "file\\_name"
    assert _escape_like("it's 100% done_now") == "it''s 100\\% done\\_now"


def test_escape():
    assert _escape("hello") == "hello"
    assert _escape("it's") == "it''s"


def test_hybrid_search():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.add_chunks("/test", [_make_chunk()])

        results = store.search_hybrid("/test", [0.1] * DIM, "test_func", limit=5)
        assert len(results) >= 1
        assert results[0].name == "test_func"


def test_file_pattern_filter():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.add_chunks("/test", [_make_chunk("/test/src/app.py", "app_func")])
        store.add_chunks("/test", [_make_chunk("/test/lib/util.py", "util_func")])

        results = store.search_vector(
            "/test", [0.1] * DIM, limit=10, file_pattern="src"
        )
        file_paths = [r.file_path for r in results]
        assert all("src" in fp for fp in file_paths)


def test_language_filter():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        py_chunk = _make_chunk("/test/a.py", "py_func")
        py_chunk["language"] = "python"
        js_chunk = _make_chunk("/test/b.js", "js_func")
        js_chunk["language"] = "javascript"
        store.add_chunks("/test", [py_chunk, js_chunk])

        results = store.search_vector("/test", [0.1] * DIM, limit=10, language="python")
        assert all(r.language == "python" for r in results)


def test_delete_project():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.add_chunks("/test", [_make_chunk()])

        hashes_before = store.get_file_hashes("/test")
        assert len(hashes_before) > 0

        store.delete_project("/test")

        db = store._get_db("/test")
        assert TABLE_NAME not in db.list_tables()
        assert "/test" not in store._tables


def test_save_meta():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.save_meta("/test", 10, "test-model")

        meta = store._read_meta("/test")
        assert meta["total_files"] == 10
        assert meta["embedding_model"] == "test-model"
        assert meta["embedding_dimension"] == DIM


def test_row_to_result_snippet_truncation():
    content = "\n".join(f"line {i}" for i in range(100))
    row = {
        "file_path": "/test.py",
        "name": "func",
        "chunk_type": "function",
        "start_line": 1,
        "end_line": 100,
        "content": content,
        "language": "python",
        "_distance": 0.5,
    }
    result = _row_to_result(row, snippet_max_lines=10)
    assert "... (90 more lines)" in result.snippet


def test_store_dimension_property():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp, dimension=512)
        assert store.dimension == 512
