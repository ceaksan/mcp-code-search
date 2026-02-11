"""File change detection for incremental indexing."""

from __future__ import annotations

import os
from pathlib import Path


def compute_file_hash(file_path: str) -> str:
    """Compute a hash based on mtime + size for change detection."""
    try:
        stat = os.stat(file_path)
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return ""


def get_changed_files(
    file_paths: list[str],
    stored_hashes: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Determine which files need re-indexing.

    Returns (changed_files, deleted_files).
    - changed_files: new or modified files
    - deleted_files: files in stored_hashes but no longer on disk
    """
    current_files = set(file_paths)
    stored_files = set(stored_hashes.keys())

    deleted = list(stored_files - current_files)

    changed = []
    for fp in file_paths:
        current_hash = compute_file_hash(fp)
        stored_hash = stored_hashes.get(fp, "")
        if current_hash != stored_hash:
            changed.append(fp)

    return changed, deleted
