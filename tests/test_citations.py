"""
test_citations.py — Unit tests for citation extraction and validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock

from src.citations import (
    Citation,
    CitationResult,
    extract_and_validate_citations,
    format_citations_for_display,
)
from src.vector_store import SearchResult


def make_result(chunk_id: str, text: str, score: float = 0.85, filename: str = "rama.txt") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_filename": filename, "chunk_id": chunk_id},
        distance=1.0 - score,
        rank=1,
    )


class TestExtractAndValidateCitations:
    def test_valid_citations_extracted(self):
        answer = "Rama was exiled [Source 1] and Hanuman helped [Source 2]."
        chunks = [
            make_result("c001", "Rama was exiled from Ayodhya."),
            make_result("c002", "Hanuman helped Rama find Sita."),
        ]
        result = extract_and_validate_citations(answer, chunks)
        assert len(result.citations) == 2
        assert all(c.is_valid for c in result.citations)

    def test_invalid_citation_detected(self):
        answer = "Some claim [Source 5]."
        chunks = [make_result("c001", "Only one chunk available.")]
        result = extract_and_validate_citations(answer, chunks)
        invalid = [c for c in result.citations if not c.is_valid]
        assert len(invalid) == 1
        assert invalid[0].source_number == 5

    def test_insufficient_evidence_detected(self):
        answer = "Not found. [INSUFFICIENT_EVIDENCE]"
        result = extract_and_validate_citations(answer, [])
        assert result.has_insufficient_evidence

    def test_clean_answer_strips_marker(self):
        answer = "Info not found. [INSUFFICIENT_EVIDENCE]"
        result = extract_and_validate_citations(answer, [])
        assert "[INSUFFICIENT_EVIDENCE]" not in result.clean_answer

    def test_implicit_citations_added_when_none_cited(self):
        answer = "Hanuman was a devoted ally of Rama."
        chunks = [
            make_result("c001", "Hanuman served Rama faithfully."),
            make_result("c002", "Rama trusted Hanuman completely."),
        ]
        result = extract_and_validate_citations(answer, chunks)
        # Should still add implicit citations from retrieved chunks
        assert len(result.citations) > 0

    def test_citation_validity_rate(self):
        answer = "Claim [Source 1]. Invalid [Source 9]."
        chunks = [make_result("c001", "Real chunk.")]
        result = extract_and_validate_citations(answer, chunks)
        # 1 valid, 1 invalid → rate < 1
        assert result.citation_validity_rate < 1.0

    def test_no_answer_no_chunks(self):
        result = extract_and_validate_citations("", [])
        assert not result.has_citations or all(not c.is_valid for c in result.citations)

    def test_duplicate_citation_numbers_handled(self):
        answer = "Fact [Source 1]. Another [Source 1]."
        chunks = [make_result("c001", "Some text.")]
        result = extract_and_validate_citations(answer, chunks)
        # Should not produce duplicate citations for the same number
        numbers = [c.source_number for c in result.citations]
        assert len(numbers) == len(set(numbers))


class TestFormatCitationsForDisplay:
    def test_returns_list_of_dicts(self):
        answer = "Fact [Source 1]."
        chunks = [make_result("c001", "Supporting text.")]
        citation_result = extract_and_validate_citations(answer, chunks)
        display = format_citations_for_display(citation_result)
        assert isinstance(display, list)
        if display:
            assert "label" in display[0]
            assert "excerpt" in display[0]

    def test_invalid_citations_excluded(self):
        answer = "Fact [Source 99]."
        chunks = [make_result("c001", "Real chunk.")]
        citation_result = extract_and_validate_citations(answer, chunks)
        display = format_citations_for_display(citation_result)
        for item in display:
            assert item["valid"]
