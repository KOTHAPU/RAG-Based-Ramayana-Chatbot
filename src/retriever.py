"""
retriever.py — Semantic retrieval from the ChromaDB vector store.

Supports:
- Dense (semantic) retrieval using the embedding model.
- Optional similarity threshold filtering.
- Optional deduplication of near-identical passages.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.embeddings import EmbeddingModel
from src.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Dense retrieval from a VectorStore using an EmbeddingModel.

    Usage::

        retriever = Retriever(vector_store, embedding_model)
        results = retriever.retrieve("Who is Hanuman?", top_k=5)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.3,
        dedup_threshold: float = 0.97,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.
            similarity_threshold: Override for the instance threshold.

        Returns:
            List of SearchResult objects sorted by descending similarity.
        """
        if not query.strip():
            logger.warning("Empty query received by retriever.")
            return []

        if self.vector_store.is_empty():
            logger.warning("Vector store is empty; cannot retrieve.")
            return []

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        # Embed the query
        try:
            query_embedding = self.embedding_model.embed_query(query)
        except Exception as exc:
            logger.error("Failed to embed query: %s", exc)
            return []

        # Retrieve from ChromaDB (fetch extra to allow filtering)
        raw_results = self.vector_store.query(query_embedding, top_k=top_k * 2)

        # Filter by similarity threshold
        filtered = [r for r in raw_results if r.similarity_score >= threshold]

        # Deduplicate near-identical passages
        filtered = self._deduplicate(filtered)

        # Return top_k
        final = filtered[:top_k]

        logger.info(
            "Query '%s…' → %d results (threshold=%.2f, raw=%d).",
            query[:60],
            len(final),
            threshold,
            len(raw_results),
        )
        return final

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove near-duplicate passages based on text overlap."""
        if len(results) <= 1:
            return results

        seen_texts: list[str] = []
        unique: list[SearchResult] = []

        for result in results:
            is_dup = False
            for seen in seen_texts:
                if _text_overlap(result.text, seen) >= self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(result)
                seen_texts.append(result.text)

        return unique


def _text_overlap(a: str, b: str) -> float:
    """
    Compute character-level Jaccard similarity between two strings.
    Used for near-duplicate detection.
    """
    if not a or not b:
        return 0.0
    # Use word-level sets for a faster approximation
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0
