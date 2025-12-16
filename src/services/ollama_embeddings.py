"""Client for Ollama embeddings."""

import math
import time
from typing import List

import requests

from src.utils.config import get_settings
from src.utils.logger import get_logger


class OllamaEmbeddingError(Exception):
    """Exception raised when Ollama embedding operations fail."""


class OllamaEmbeddingClient:
    """Client for generating embeddings using Ollama."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        settings = get_settings()
        self.model_name = model_name
        self.base_url = base_url
        self._logger = get_logger("audra.services.ollama_embeddings")

        # Ollama embeddings endpoint
        self.embeddings_url = f"{self.base_url}/api/embeddings"

        # Test connection
        try:
            response = requests.post(
                self.embeddings_url,
                json={"model": self.model_name, "prompt": "test"},
                timeout=10,
            )
            response.raise_for_status()
            self._logger.info(
                "Ollama embedding client initialized.",
                extra={"context": {"model": self.model_name, "base_url": self.base_url}},
            )
        except requests.exceptions.RequestException as exc:
            self._logger.error(
                "Failed to connect to Ollama embeddings.",
                extra={"context": {"error": str(exc), "url": self.embeddings_url}},
            )
            raise OllamaEmbeddingError(f"Failed to initialize Ollama embedding client: {exc}") from exc

    def embed_text(self, text: str, prefix: str = "") -> List[float]:
        """Generate an embedding for the provided text."""
        if not text:
            raise ValueError("Text must be a non-empty string.")

        prompt = f"{prefix}{text}" if prefix else text
        start_time = time.perf_counter()

        try:
            response = requests.post(
                self.embeddings_url,
                json={"model": self.model_name, "prompt": prompt},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "embedding" not in data:
                raise OllamaEmbeddingError("No embedding returned from Ollama")

            embedding = data["embedding"]
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            self._logger.debug(
                "Embedding generated.",
                extra={
                    "context": {
                        "operation": "embed_text",
                        "latency_ms": latency_ms,
                        "text_length": len(text),
                        "embedding_dim": len(embedding),
                    }
                },
            )

            return self._normalize(embedding)

        except requests.exceptions.RequestException as exc:
            self._logger.error(
                "Ollama embedding request failed.",
                extra={"context": {"error": str(exc), "text_preview": text[:100]}},
            )
            raise OllamaEmbeddingError(f"Failed to generate embedding: {exc}") from exc

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed a list of texts."""
        if not texts:
            return []

        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))

        return embeddings

    def get_query_embedding(self, query: str) -> List[float]:
        """Return an embedding configured for query semantics."""
        return self.embed_text(query, prefix="search_query: ")

    def get_document_embedding(self, text: str) -> List[float]:
        """Return an embedding configured for document storage semantics."""
        return self.embed_text(text, prefix="search_document: ")

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
