"""
test_document_loader.py — Unit tests for document loading and metadata preservation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os

from src.document_loader import Document, load_txt, load_markdown, load_documents
from src.utils import clean_text


# ── Text cleaning ─────────────────────────────────────────────────────────────

class TestCleanText:
    def test_normalises_whitespace(self):
        result = clean_text("Rama   went   to   the   forest.")
        assert "   " not in result

    def test_preserves_paragraph_breaks(self):
        result = clean_text("First paragraph.\n\nSecond paragraph.")
        assert "\n\n" in result

    def test_strips_control_characters(self):
        result = clean_text("Rama\x00Sita")
        assert "\x00" not in result

    def test_replaces_smart_quotes(self):
        result = clean_text("\u201cRama\u201d said \u2018Sita\u2019")
        assert '"Rama"' in result
        assert "'Sita'" in result

    def test_preserves_sanskrit_names(self):
        names = ["Rama", "Sita", "Lakshmana", "Hanuman", "Ravana", "Valmiki"]
        text = " ".join(names)
        result = clean_text(text)
        for name in names:
            assert name in result


# ── TXT loading ───────────────────────────────────────────────────────────────

class TestLoadTxt:
    def test_loads_utf8_file(self, tmp_path):
        f = tmp_path / "rama.txt"
        f.write_text("Rama was the prince of Ayodhya.", encoding="utf-8")
        docs = load_txt(f)
        assert len(docs) == 1
        assert "Ayodhya" in docs[0].text

    def test_metadata_preserved(self, tmp_path):
        f = tmp_path / "story.txt"
        f.write_text("Sita was the daughter of King Janaka." * 5, encoding="utf-8")
        docs = load_txt(f)
        assert docs[0].metadata["source_filename"] == "story.txt"
        assert docs[0].metadata["file_type"] == "txt"
        assert "file_hash" in docs[0].metadata

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   ", encoding="utf-8")
        docs = load_txt(f)
        assert docs == []

    def test_document_is_not_empty(self, tmp_path):
        f = tmp_path / "content.txt"
        content = "Hanuman was the devoted servant of Rama. " * 10
        f.write_text(content, encoding="utf-8")
        docs = load_txt(f)
        assert not docs[0].is_empty


# ── Markdown loading ──────────────────────────────────────────────────────────

class TestLoadMarkdown:
    def test_splits_on_headings(self, tmp_path):
        content = "# Ayodhya Kanda\n\nRama was exiled.\n\n## The Exile\n\nDetails of exile."
        f = tmp_path / "kanda.md"
        f.write_text(content, encoding="utf-8")
        docs = load_markdown(f)
        assert len(docs) >= 1

    def test_chapter_metadata_extracted(self, tmp_path):
        content = "# Bala Kanda\n\nThis is the first kanda."
        f = tmp_path / "bala.md"
        f.write_text(content, encoding="utf-8")
        docs = load_markdown(f)
        assert any("Bala Kanda" in d.metadata.get("chapter", "") for d in docs)

    def test_file_type_is_markdown(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Test\n\nSome content about Rama.", encoding="utf-8")
        docs = load_markdown(f)
        for doc in docs:
            assert doc.metadata["file_type"] == "markdown"


# ── Directory loading ─────────────────────────────────────────────────────────

class TestLoadDocuments:
    def test_loads_multiple_files(self, tmp_path):
        (tmp_path / "file1.txt").write_text("Rama the hero of Ayodhya." * 5, encoding="utf-8")
        (tmp_path / "file2.txt").write_text("Sita the princess of Mithila." * 5, encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) >= 2

    def test_skips_duplicate_content(self, tmp_path):
        content = "Rama the hero of Ayodhya and the dharmic king." * 5
        (tmp_path / "dup1.txt").write_text(content, encoding="utf-8")
        (tmp_path / "dup2.txt").write_text(content, encoding="utf-8")
        docs = load_documents(tmp_path)
        # Should only keep one of the duplicates
        texts = [d.text for d in docs]
        assert len(set(texts)) == len(texts)

    def test_skips_already_indexed(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("Content about Rama." * 5, encoding="utf-8")
        from src.utils import file_hash
        fhash = file_hash(f)
        docs = load_documents(tmp_path, skip_hashes={fhash})
        assert docs == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent"
        docs = load_documents(missing)
        assert docs == []

    def test_unsupported_extension_skipped(self, tmp_path):
        (tmp_path / "data.csv").write_text("col1,col2\n1,2", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert docs == []
