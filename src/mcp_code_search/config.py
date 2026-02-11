"""Configuration management."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_IGNORE_PATTERNS = [
    "node_modules",
    ".git",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    ".nuxt",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "uv.lock",
    "poetry.lock",
    ".DS_Store",
]

GLOBAL_CONFIG_PATH = Path.home() / ".config" / "mcp-code-search" / "config.toml"
DEFAULT_STORAGE_PATH = Path.home() / ".local" / "share" / "mcp-code-search"
PROJECT_CONFIG_NAME = ".code-search.toml"


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"
    model: str = "jinaai/jina-embeddings-v2-base-code"
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class IndexingConfig:
    ignore_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS)
    )
    max_file_size_kb: int = 500
    chunk_overlap_lines: int = 2
    batch_size: int = 32
    max_chunk_lines: int = 50


@dataclass
class SearchConfig:
    rrf_k: int = 60
    snippet_max_lines: int = 30


@dataclass
class StorageConfig:
    base_path: Path = field(default_factory=lambda: DEFAULT_STORAGE_PATH)


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def load(cls, project_path: Optional[Path] = None) -> Config:
        config = cls()

        if GLOBAL_CONFIG_PATH.exists():
            _merge_toml(config, GLOBAL_CONFIG_PATH)

        if project_path:
            project_config = Path(project_path) / PROJECT_CONFIG_NAME
            if project_config.exists():
                _merge_toml(config, project_config)

        return config

    def get_project_storage_path(self, project_path: str) -> Path:
        project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:16]
        name = Path(project_path).name
        dir_name = f"{name}-{project_hash}"
        path = self.storage.base_path / "projects" / dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path


def _merge_toml(config: Config, path: Path) -> None:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if "embedding" in data:
        emb = data["embedding"]
        if "provider" in emb:
            config.embedding.provider = emb["provider"]
        if "model" in emb:
            config.embedding.model = emb["model"]
        if "ollama" in emb:
            ol = emb["ollama"]
            if "model" in ol:
                config.embedding.ollama_model = ol["model"]
            if "base_url" in ol:
                config.embedding.ollama_base_url = ol["base_url"]

    if "indexing" in data:
        idx = data["indexing"]
        if "ignore_patterns" in idx:
            config.indexing.ignore_patterns = idx["ignore_patterns"]
        if "max_file_size_kb" in idx:
            config.indexing.max_file_size_kb = idx["max_file_size_kb"]
        if "chunk_overlap_lines" in idx:
            config.indexing.chunk_overlap_lines = idx["chunk_overlap_lines"]
        if "batch_size" in idx:
            config.indexing.batch_size = idx["batch_size"]
        if "max_chunk_lines" in idx:
            config.indexing.max_chunk_lines = idx["max_chunk_lines"]

    if "search" in data:
        srch = data["search"]
        if "rrf_k" in srch:
            config.search.rrf_k = srch["rrf_k"]
        if "snippet_max_lines" in srch:
            config.search.snippet_max_lines = srch["snippet_max_lines"]

    if "storage" in data:
        st = data["storage"]
        if "base_path" in st:
            config.storage.base_path = Path(st["base_path"]).expanduser()
