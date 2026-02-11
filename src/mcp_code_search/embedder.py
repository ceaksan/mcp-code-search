"""Embedding generation with sentence-transformers (primary) and Ollama (optional)."""

from __future__ import annotations

import logging
from typing import Protocol

from mcp_code_search.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Embedding provider interface."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerEmbedder:
    """sentence-transformers based embedder with lazy loading."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        self._load()
        return self._dimension


class OllamaEmbedder:
    """Ollama-based embedder."""

    def __init__(
        self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"
    ):
        self._model = model
        self._base_url = base_url
        self._dimension: int | None = None
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise RuntimeError(
                    "ollama package not installed. Install with: pip install ollama"
                )
            self._client = ollama.Client(host=self._base_url)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        results = []
        for text in texts:
            resp = client.embed(model=self._model, input=text)
            embedding = resp["embeddings"][0]
            results.append(embedding)
            if self._dimension is None:
                self._dimension = len(embedding)
        return results

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Get dimension by embedding a test string
            result = self.embed(["test"])
            self._dimension = len(result[0])
        return self._dimension


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Create an embedder based on config."""
    if config.provider == "ollama":
        return OllamaEmbedder(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
        )
    return SentenceTransformerEmbedder(model_name=config.model)
