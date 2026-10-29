"""
text_splitter.py — Split documents into overlapping chunks for embedding.

Uses a token-aware sliding window with configurable size and overlap.
Preserves paragraph boundaries where possible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from src.document_loader import Document
from src.utils import sanitise_metadata, text_hash, try_count_tokens

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    A text chunk ready for embedding and storage.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The chunk text.
        metadata: Full metadata dict (ChromaDB-compatible types only).
        token_count: Estimated token count.
    """

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = try_count_tokens(self.text)

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, tokens={self.token_count}, "
            f"source={self.metadata.get('source_filename', '?')!r})"
        )


# ── Sentence / paragraph splitter ────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving verse numbers and Sanskrit terms."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    # Also split on paragraph breaks
    result: list[str] = []
    for sent in sentences:
        parts = re.split(r"\n{2,}", sent)
        result.extend(p.strip() for p in parts if p.strip())
    return result


# ── Core splitter ─────────────────────────────────────────────────────────────

class TextSplitter:
    """
    Splits Document objects into overlapping Chunk objects.

    Strategy:
    1. Split the document text into sentences / paragraphs.
    2. Build chunks by accumulating sentences until chunk_size tokens is reached.
    3. Back-fill overlap tokens from the previous chunk.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, doc: Document) -> list[Chunk]:
        """Split a single Document into Chunks."""
        sentences = _split_into_sentences(doc.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens: int = 0

        def _flush(chunk_index: int) -> None:
            nonlocal current_sentences, current_tokens
            if not current_sentences:
                return
            chunk_text = " ".join(current_sentences)
            chunk_id = f"{doc.doc_id}_c{chunk_index:04d}"
            meta = sanitise_metadata(
                {
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "doc_id": doc.doc_id,
                    "chunk_index": chunk_index,
                    "chunk_text_preview": chunk_text[:120],
                }
            )
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))

        for sent in sentences:
            sent_tokens = try_count_tokens(sent)
            # If a single sentence exceeds chunk_size, hard-split it
            if sent_tokens > self.chunk_size:
                _flush(len(chunks))
                current_sentences, current_tokens = [], 0
                # Hard split on words
                words = sent.split()
                word_buffer: list[str] = []
                word_tokens = 0
                for word in words:
                    wt = try_count_tokens(word)
                    if word_tokens + wt > self.chunk_size and word_buffer:
                        chunk_text = " ".join(word_buffer)
                        chunk_id = f"{doc.doc_id}_c{len(chunks):04d}"
                        meta = sanitise_metadata(
                            {
                                **doc.metadata,
                                "chunk_id": chunk_id,
                                "doc_id": doc.doc_id,
                                "chunk_index": len(chunks),
                                "chunk_text_preview": chunk_text[:120],
                            }
                        )
                        chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))
                        # Overlap: keep last overlap_tokens worth of words
                        word_buffer, word_tokens = _overlap_words(word_buffer, self.chunk_overlap)
                    word_buffer.append(word)
                    word_tokens += wt
                if word_buffer:
                    current_sentences = [" ".join(word_buffer)]
                    current_tokens = word_tokens
                continue

            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                _flush(len(chunks))
                # Carry over overlap
                current_sentences, current_tokens = _overlap_sentences(
                    current_sentences, self.chunk_overlap
                )

            current_sentences.append(sent)
            current_tokens += sent_tokens

        _flush(len(chunks))
        logger.debug(
            "Split '%s' into %d chunks.", doc.metadata.get("source_filename", doc.doc_id), len(chunks)
        )
        return chunks

    def split_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        """Split a list of Documents into Chunks."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        logger.info("Total chunks produced: %d from %d documents.", len(all_chunks), len(documents))
        return all_chunks


# ── Overlap helpers ───────────────────────────────────────────────────────────

def _overlap_sentences(sentences: list[str], target_tokens: int) -> tuple[list[str], int]:
    """Return the last N sentences that fit within target_tokens."""
    if not sentences:
        return [], 0
    selected: list[str] = []
    total = 0
    for sent in reversed(sentences):
        t = try_count_tokens(sent)
        if total + t > target_tokens:
            break
        selected.insert(0, sent)
        total += t
    return selected, total


def _overlap_words(words: list[str], target_tokens: int) -> tuple[list[str], int]:
    """Return the last N words that fit within target_tokens."""
    selected: list[str] = []
    total = 0
    for word in reversed(words):
        t = try_count_tokens(word)
        if total + t > target_tokens:
            break
        selected.insert(0, word)
        total += t
    return selected, total
