"""
embeddings.py — Sentence-Transformer embedding model wrapper.

Downloads BAAI/bge-small-en-v1.5 (~130 MB) on first use and caches it locally.
The same model must be used for both ingestion and retrieval.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

logger = logging.getLogger(__name__)

# Type alias
EmbeddingVector = list[float]


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for producing dense embeddings.

    The model is loaded lazily on first call to avoid blocking the import.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded.")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {exc}") from exc

    def embed_query(self, query: str) -> EmbeddingVector:
        """Embed a single query string (used at retrieval time)."""
        self._load()
        # BGE models benefit from a query prefix
        prefixed = f"Represent this sentence: {query}"
        embedding = self._model.encode(prefixed, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[EmbeddingVector]:
        """
        Embed a list of texts (used at ingestion time).
        BGE models use a passage prefix for document chunks.
        """
        self._load()
        prefixed = [f"Represent this passage: {t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(prefixed) > 50,
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def is_loaded(self) -> bool:
        return self._model is not None


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingModel:
    """Return a cached EmbeddingModel instance."""
    return EmbeddingModel(model_name)
