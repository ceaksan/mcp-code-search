"""Tests for the watcher module."""

import tempfile
import time
from pathlib import Path

from mcp_code_search.watcher import compute_file_hash, get_changed_files


def test_compute_file_hash():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("hello world")
        f.flush()
        h = compute_file_hash(f.name)
        assert h != ""
        assert ":" in h


def test_nonexistent_file():
    h = compute_file_hash("/nonexistent/file.py")
    assert h == ""


def test_changed_file_detection():
    with tempfile.TemporaryDirectory() as tmp:
        fp = str(Path(tmp) / "test.py")
        Path(fp).write_text("v1")
        hash_v1 = compute_file_hash(fp)

        time.sleep(0.05)
        Path(fp).write_text("v2 updated content")
        hash_v2 = compute_file_hash(fp)

        assert hash_v1 != hash_v2


def test_get_changed_files_new_file():
    with tempfile.TemporaryDirectory() as tmp:
        fp = str(Path(tmp) / "new.py")
        Path(fp).write_text("new file")

        changed, deleted = get_changed_files([fp], {})
        assert fp in changed
        assert len(deleted) == 0


def test_get_changed_files_deleted():
    changed, deleted = get_changed_files(
        [],
        {"/old/file.py": "somehash"},
    )
    assert len(changed) == 0
    assert "/old/file.py" in deleted


def test_get_changed_files_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        fp = str(Path(tmp) / "same.py")
        Path(fp).write_text("content")
        h = compute_file_hash(fp)

        changed, deleted = get_changed_files([fp], {fp: h})
        assert len(changed) == 0
        assert len(deleted) == 0
