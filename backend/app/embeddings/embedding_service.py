from __future__ import annotations

import logging
import time
from typing import Optional, Protocol, Sequence , Any

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thin, validated wrapper around an embedding model."""

    def __init__(
        self,
        embedder: Any,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if embedder is None:
            raise ValueError("embedder must not be None.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.embedder = embedder
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._dimension: Optional[int] = None

    @classmethod
    def from_model_name(cls, model_name: str, **kwargs) -> "EmbeddingService":
        """Load a SentenceTransformer model by name and wrap it."""
        from sentence_transformers import SentenceTransformer

        return cls(SentenceTransformer(model_name), **kwargs)

    @property
    def dimension(self) -> int:
        """Embedding size, detected once from the model."""
        if self._dimension is None:
            self._dimension = len(self.generate_query_embedding("dimension probe"))
        return self._dimension

    def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple document chunks."""
        if (
            not texts
            or isinstance(texts, str)
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise ValueError("texts must be a non-empty sequence of non-empty strings.")

        start_time = time.perf_counter()
        try:
            embeddings = self.embedder.encode_document(
                list(texts),
                normalize_embeddings=self.normalize_embeddings,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception:
            logger.exception("Failed to generate embeddings for %d texts", len(texts))
            raise

        results = embeddings.tolist()
        logger.info(
            "Generated %d embeddings in %.3f seconds",
            len(results),
            time.perf_counter() - start_time,
        )
        return results

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate an embedding for a single user query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        start_time = time.perf_counter()
        try:
            embedding = self.embedder.encode_query(
                query,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception:
            logger.exception("Failed to generate query embedding")
            raise

        result = embedding.tolist()
        logger.info(
            "Generated query embedding in %.3f seconds",
            time.perf_counter() - start_time,
        )
        return result