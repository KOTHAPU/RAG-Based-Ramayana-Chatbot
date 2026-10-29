"""
citations.py — Extract, validate, and format source citations from LLM answers.

After generation, cross-references [Source N] markers in the answer against
the actually retrieved chunks to prevent hallucinated citations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.vector_store import SearchResult
from src.utils import format_source_label, truncate_text

logger = logging.getLogger(__name__)

# Pattern to match [Source N] citations in generated text
_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

# Marker for insufficient evidence
_INSUFFICIENT_MARKER = "[INSUFFICIENT_EVIDENCE]"


@dataclass
class Citation:
    """A validated citation extracted from the LLM answer."""

    source_number: int
    chunk_id: str
    source_label: str
    excerpt: str
    similarity_score: float
    metadata: dict[str, Any]
    is_valid: bool = True  # False if the source number has no matching chunk

    def to_dict(self) -> dict:
        return {
            "source_number": self.source_number,
            "chunk_id": self.chunk_id,
            "source_label": self.source_label,
            "excerpt": self.excerpt,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
        }


@dataclass
class CitationResult:
    """Result of citation extraction from a generated answer."""

    raw_answer: str
    clean_answer: str
    citations: list[Citation]
    has_insufficient_evidence: bool
    citation_validity_rate: float

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    @property
    def valid_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.is_valid]


def extract_and_validate_citations(
    answer: str,
    retrieved_chunks: list[SearchResult],
    excerpt_max_chars: int = 350,
) -> CitationResult:
    """
    Extract [Source N] references from the answer and validate them against
    the actually retrieved chunks.

    Args:
        answer: Raw LLM-generated answer text.
        retrieved_chunks: The SearchResult list used for this generation.
        excerpt_max_chars: Maximum excerpt length for display.

    Returns:
        CitationResult with validated citations.
    """
    has_insufficient = _INSUFFICIENT_MARKER in answer
    clean_answer = answer.replace(_INSUFFICIENT_MARKER, "").strip()

    # Find all [Source N] references in the answer
    cited_numbers = set(int(m) for m in _CITATION_PATTERN.findall(answer))

    citations: list[Citation] = []
    seen_numbers: set[int] = set()

    for num in sorted(cited_numbers):
        if num in seen_numbers:
            continue
        seen_numbers.add(num)

        # Source numbers are 1-indexed in our prompts
        chunk_index = num - 1

        if 0 <= chunk_index < len(retrieved_chunks):
            chunk = retrieved_chunks[chunk_index]
            citations.append(
                Citation(
                    source_number=num,
                    chunk_id=chunk.chunk_id,
                    source_label=format_source_label(chunk.metadata),
                    excerpt=truncate_text(chunk.text, excerpt_max_chars),
                    similarity_score=chunk.similarity_score,
                    metadata=chunk.metadata,
                    is_valid=True,
                )
            )
        else:
            # Citation references a non-existent source — invalid
            logger.warning(
                "Answer cited [Source %d] but only %d chunks were retrieved.",
                num,
                len(retrieved_chunks),
            )
            citations.append(
                Citation(
                    source_number=num,
                    chunk_id="",
                    source_label=f"Source {num} (not found in retrieved chunks)",
                    excerpt="",
                    similarity_score=0.0,
                    metadata={},
                    is_valid=False,
                )
            )

    # Also include retrieved chunks that weren't cited (for the "Sources Used" panel)
    # but only list the ones that were actually used (cited)
    if not citations and retrieved_chunks:
        # If the model produced an answer but cited nothing, include top-3 as implicit
        for chunk in retrieved_chunks[:3]:
            citations.append(
                Citation(
                    source_number=0,
                    chunk_id=chunk.chunk_id,
                    source_label=format_source_label(chunk.metadata),
                    excerpt=truncate_text(chunk.text, excerpt_max_chars),
                    similarity_score=chunk.similarity_score,
                    metadata=chunk.metadata,
                    is_valid=True,
                )
            )

    validity_rate = (
        sum(1 for c in citations if c.is_valid) / len(citations) if citations else 1.0
    )

    return CitationResult(
        raw_answer=answer,
        clean_answer=clean_answer,
        citations=citations,
        has_insufficient_evidence=has_insufficient,
        citation_validity_rate=validity_rate,
    )


def format_citations_for_display(citation_result: CitationResult) -> list[dict]:
    """Convert citations to a list of dicts suitable for Streamlit display."""
    return [
        {
            "number": c.source_number,
            "label": c.source_label,
            "excerpt": c.excerpt,
            "score": f"{c.similarity_score:.2f}",
            "metadata": c.metadata,
            "valid": c.is_valid,
        }
        for c in citation_result.citations
        if c.is_valid
    ]
