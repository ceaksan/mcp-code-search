"""Data models for code search."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CodeChunk(BaseModel):
    """A chunk of code stored in LanceDB."""

    file_path: str
    project_path: str
    language: str = ""
    chunk_type: str = ""  # function, class, method, block, module
    name: str = ""  # function/class name
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    file_hash: str = ""  # mtime+size hash for incremental indexing


class SearchResult(BaseModel):
    """A single search result."""

    file_path: str
    name: str = ""
    chunk_type: str = ""
    start_line: int = 0
    end_line: int = 0
    score: float = 0.0
    snippet: str = ""
    language: str = ""


class IndexStatus(BaseModel):
    """Status of a project index."""

    project_path: str
    total_files: int = 0
    total_chunks: int = 0
    last_indexed: Optional[datetime] = None
    embedding_model: str = ""


class ProjectInfo(BaseModel):
    """Info about an indexed project."""

    project_path: str
    storage_path: str
    total_chunks: int = 0
    last_indexed: Optional[datetime] = None
