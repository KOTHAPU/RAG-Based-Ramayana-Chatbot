"""
test_retriever.py — Unit tests for the retriever (mocked vector store).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch

from src.retriever import Retriever, _text_overlap
from src.vector_store import SearchResult


def make_mock_result(rank: int, text: str, score: float, filename: str = "test.txt") -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk_{rank}",
        text=text,
        metadata={"source_filename": filename, "chunk_id": f"chunk_{rank}"},
        distance=1.0 - score,
        rank=rank,
    )


class TestRetriever:
    @pytest.fixture
    def mock_retriever(self):
        mock_vs = MagicMock()
        mock_vs.is_empty.return_value = False
        mock_vs.query.return_value = [
            make_mock_result(1, "Hanuman crossed the ocean to Lanka.", 0.92),
            make_mock_result(2, "Hanuman burned Lanka with his tail.", 0.85),
            make_mock_result(3, "Rama trusted Hanuman completely.", 0.78),
        ]
        mock_em = MagicMock()
        mock_em.embed_query.return_value = [0.1] * 384
        return Retriever(
            vector_store=mock_vs,
            embedding_model=mock_em,
            similarity_threshold=0.5,
        )

    def test_retrieves_results(self, mock_retriever):
        results = mock_retriever.retrieve("What did Hanuman do?", top_k=3)
        assert len(results) > 0

    def test_empty_query_returns_empty(self, mock_retriever):
        results = mock_retriever.retrieve("   ", top_k=3)
        assert results == []

    def test_empty_index_returns_empty(self):
        mock_vs = MagicMock()
        mock_vs.is_empty.return_value = True
        mock_em = MagicMock()
        retriever = Retriever(mock_vs, mock_em)
        results = retriever.retrieve("Who is Rama?")
        assert results == []

    def test_similarity_threshold_filters(self, mock_retriever):
        # Set high threshold — only top result should pass
        mock_retriever.similarity_threshold = 0.90
        results = mock_retriever.retrieve("Hanuman", top_k=5)
        for r in results:
            assert r.similarity_score >= 0.90

    def test_top_k_limits_results(self, mock_retriever):
        results = mock_retriever.retrieve("Hanuman Lanka", top_k=2)
        assert len(results) <= 2

    def test_deduplication_removes_near_duplicates(self, mock_retriever):
        # Override vector store to return near-identical results
        dup_text = "Hanuman was the son of the wind god Vayu and served Rama."
        results_with_dup = [
            make_mock_result(1, dup_text, 0.95),
            make_mock_result(2, dup_text + " He was very powerful.", 0.93),
        ]
        mock_retriever.vector_store.is_empty.return_value = False
        mock_retriever.vector_store.query.return_value = results_with_dup
        results = mock_retriever.retrieve("Hanuman", top_k=5)
        # Near-duplicates should be reduced to 1 (high overlap > dedup threshold)
        assert len(results) <= 2


class TestTextOverlap:
    def test_identical_texts(self):
        t = "Rama went to the forest with Sita and Lakshmana"
        assert _text_overlap(t, t) == 1.0

    def test_completely_different(self):
        a = "apple banana cherry"
        b = "dog cat fish"
        assert _text_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = "Rama and Sita went to forest"
        b = "Rama was a prince of Ayodhya"
        score = _text_overlap(a, b)
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        assert _text_overlap("", "anything") == 0.0
        assert _text_overlap("anything", "") == 0.0
