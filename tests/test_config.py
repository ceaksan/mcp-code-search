"""Tests for the config module."""

import tempfile
from pathlib import Path

from mcp_code_search.config import Config, SearchConfig, _merge_toml


def test_default_config():
    config = Config()
    assert config.embedding.provider == "sentence-transformers"
    assert config.embedding.model == "jinaai/jina-embeddings-v2-base-code"
    assert config.indexing.max_file_size_kb == 500
    assert config.indexing.batch_size == 32
    assert config.indexing.max_chunk_lines == 50
    assert config.search.rrf_k == 60
    assert config.search.snippet_max_lines == 30


def test_toml_merge():
    config = Config()
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("""
[embedding]
model = "custom-model"

[indexing]
batch_size = 64
max_chunk_lines = 100

[search]
rrf_k = 30
snippet_max_lines = 50

[storage]
base_path = "/tmp/test-storage"
""")
        f.flush()
        _merge_toml(config, Path(f.name))

    assert config.embedding.model == "custom-model"
    assert config.indexing.batch_size == 64
    assert config.indexing.max_chunk_lines == 100
    assert config.search.rrf_k == 30
    assert config.search.snippet_max_lines == 50
    assert config.storage.base_path == Path("/tmp/test-storage")


def test_storage_path_determinism():
    config = Config()
    path1 = config.get_project_storage_path("/home/user/project")
    path2 = config.get_project_storage_path("/home/user/project")
    assert path1 == path2


def test_storage_path_different_projects():
    config = Config()
    with tempfile.TemporaryDirectory() as tmp:
        config.storage.base_path = Path(tmp)
        path1 = config.get_project_storage_path("/project/a")
        path2 = config.get_project_storage_path("/project/b")
        assert path1 != path2


def test_load_without_files():
    config = Config.load()
    assert config.embedding.provider == "sentence-transformers"


def test_search_config_defaults():
    sc = SearchConfig()
    assert sc.rrf_k == 60
    assert sc.snippet_max_lines == 30


def test_toml_merge_partial():
    config = Config()
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("""
[embedding]
provider = "ollama"
""")
        f.flush()
        _merge_toml(config, Path(f.name))

    assert config.embedding.provider == "ollama"
    assert config.embedding.model == "jinaai/jina-embeddings-v2-base-code"
    assert config.indexing.batch_size == 32
