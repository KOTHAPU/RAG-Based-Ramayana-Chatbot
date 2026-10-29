"""
test_rag_pipeline.py — Integration tests for the RAG pipeline (mocked dependencies).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.rag_pipeline import RAGPipeline, RAGResponse
from src.vector_store import SearchResult
from src.generation import GenerationResponse
from src.citations import CitationResult, Citation
from src.config import settings


def make_mock_result(rank: int = 1, text: str = "Hanuman was Rama's ally.") -> SearchResult:
    return SearchResult(
        chunk_id=f"c00{rank}",
        text=text,
        metadata={"source_filename": "ramayana.txt", "chunk_id": f"c00{rank}"},
        distance=0.1,
        rank=rank,
    )


def make_mock_pipeline(index_empty: bool = False) -> RAGPipeline:
    """Create a RAGPipeline with all dependencies mocked."""
    mock_retriever = MagicMock()
    mock_retriever.vector_store.is_empty.return_value = index_empty
    mock_retriever.vector_store.count.return_value = 0 if index_empty else 100

    if index_empty:
        mock_retriever.retrieve.return_value = []
    else:
        mock_retriever.retrieve.return_value = [make_mock_result()]

    mock_gen = MagicMock()
    mock_gen.generate.return_value = GenerationResponse(
        content="Hanuman was a mighty devotee of Rama [Source 1].",
        model="test-model",
        provider="groq",
    )

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.side_effect = lambda q, h: q  # return question unchanged
    mock_rewriter.needs_rewriting.return_value = False

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generation_client=mock_gen,
        query_rewriter=mock_rewriter,
        cfg=settings,
    )
    return pipeline


class TestRAGPipeline:
    def test_answer_returns_response(self):
        pipeline = make_mock_pipeline()
        response = pipeline.answer("Who is Hanuman?")
        assert isinstance(response, RAGResponse)
        assert response.question == "Who is Hanuman?"

    def test_empty_question_returns_error(self):
        pipeline = make_mock_pipeline()
        response = pipeline.answer("   ")
        assert response.has_error or response.is_insufficient

    def test_empty_index_returns_insufficient(self):
        pipeline = make_mock_pipeline(index_empty=True)
        response = pipeline.answer("Who is Rama?")
        assert response.is_insufficient

    def test_retrieved_chunks_in_response(self):
        pipeline = make_mock_pipeline()
        response = pipeline.answer("Who is Hanuman?")
        assert len(response.retrieved_chunks) > 0

    def test_answer_not_empty_when_results_exist(self):
        pipeline = make_mock_pipeline()
        response = pipeline.answer("Tell me about Hanuman.")
        assert response.answer.strip() != ""

    def test_query_rewriter_called_with_history(self):
        pipeline = make_mock_pipeline()
        history = [
            {"role": "user", "content": "Who is Hanuman?"},
            {"role": "assistant", "content": "Hanuman is Rama's ally."},
        ]
        pipeline.answer("What did he do in Lanka?", history=history)
        pipeline.query_rewriter.rewrite.assert_called_once()

    def test_model_and_provider_in_response(self):
        pipeline = make_mock_pipeline()
        response = pipeline.answer("Who is Sita?")
        assert response.model is not None
        assert response.provider is not None

    def test_rewritten_query_set(self):
        pipeline = make_mock_pipeline()
        pipeline.query_rewriter.rewrite.side_effect = lambda q, h: f"standalone: {q}"
        response = pipeline.answer("What did he do?", history=[{"role": "user", "content": "Who is Rama?"}])
        assert response.rewritten_query.startswith("standalone:")


class TestRAGPipelineMissingAPIKey:
    def test_generation_error_handled_gracefully(self):
        pipeline = make_mock_pipeline()
        pipeline.generation_client.generate.side_effect = RuntimeError("API key missing")
        response = pipeline.answer("Who is Rama?")
        assert response.has_error or response.is_insufficient
