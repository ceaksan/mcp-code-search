"""Tests for the embedder module."""

from unittest.mock import MagicMock, patch

from mcp_code_search.config import EmbeddingConfig
from mcp_code_search.embedder import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)


def test_sentence_transformer_embed():
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectors = emb.embed(["hello world", "def foo(): pass"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_dimension():
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    assert emb.dimension == 384


def test_create_embedder_default():
    config = EmbeddingConfig()
    emb = create_embedder(config)
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb._model_name == "all-MiniLM-L6-v2"


def test_create_embedder_explicit_model():
    config = EmbeddingConfig(model="all-MiniLM-L6-v2")
    emb = create_embedder(config)
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb._model_name == "all-MiniLM-L6-v2"


def test_batch_embedding():
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    texts = [f"text number {i}" for i in range(10)]
    vectors = emb.embed(texts)
    assert len(vectors) == 10
    for v in vectors:
        assert len(v) == 384


def test_empty_input():
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectors = emb.embed([])
    assert len(vectors) == 0


def test_single_input():
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vectors = emb.embed(["hello"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_ollama_embedder_lazy_init():
    emb = OllamaEmbedder(model="test-model", base_url="http://localhost:11434")
    assert emb._client is None

    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1] * 768]}

    mock_ollama = MagicMock()
    mock_ollama.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        emb.embed(["test"])
        assert emb._client is mock_client
        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")

        # Second call should reuse client
        emb.embed(["test2"])
        mock_ollama.Client.assert_called_once()
