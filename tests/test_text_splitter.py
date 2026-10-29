"""
test_text_splitter.py — Unit tests for the overlapping text chunker.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.document_loader import Document
from src.text_splitter import Chunk, TextSplitter


def make_document(text: str, filename: str = "test.txt") -> Document:
    return Document(
        text=text,
        metadata={"source_filename": filename, "file_type": "txt"},
    )


class TestTextSplitter:
    def test_basic_split(self):
        long_text = "Rama walked through the forest. " * 100
        doc = make_document(long_text)
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        text = "Hanuman leaped over the ocean to reach Lanka. " * 100
        doc = make_document(text)
        splitter = TextSplitter(chunk_size=80, chunk_overlap=15)
        chunks = splitter.split_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved_in_chunks(self):
        doc = make_document("Sita was found in Ashoka grove. " * 50, "sita.txt")
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata["source_filename"] == "sita.txt"
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_short_document_single_chunk(self):
        doc = make_document("Rama is the hero of the Ramayana.")
        splitter = TextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_document(doc)
        assert len(chunks) == 1

    def test_empty_document_returns_no_chunks(self):
        doc = make_document("   ")
        splitter = TextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_document(doc)
        assert chunks == []

    def test_overlap_creates_continuity(self):
        """Chunks should overlap: the end of chunk N should appear at the start of chunk N+1."""
        text = (" ".join([f"word{i}" for i in range(500)]))
        doc = make_document(text)
        splitter = TextSplitter(chunk_size=100, chunk_overlap=25)
        chunks = splitter.split_document(doc)
        if len(chunks) >= 2:
            # Last word of chunk 0 should appear somewhere in chunk 1's beginning
            last_words_c0 = set(chunks[0].text.split()[-5:])
            first_words_c1 = set(chunks[1].text.split()[:10])
            assert last_words_c0 & first_words_c1, "Expected some overlap between consecutive chunks"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextSplitter(chunk_size=100, chunk_overlap=100)

    def test_split_multiple_documents(self):
        docs = [make_document(f"Document {i} content. " * 50, f"doc{i}.txt") for i in range(3)]
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(docs)
        sources = {c.metadata["source_filename"] for c in chunks}
        assert sources == {"doc0.txt", "doc1.txt", "doc2.txt"}

    def test_token_count_populated(self):
        doc = make_document("Rama Sita Lakshmana Hanuman Ravana. " * 10)
        splitter = TextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_document(doc)
        for chunk in chunks:
            assert chunk.token_count > 0
