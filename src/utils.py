"""
utils.py — Shared utility helpers for the Ramayana RAG Chatbot.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Text helpers ──────────────────────────────────────────────────────────────

def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    # Preserve paragraph breaks (double newline)
    paragraphs = re.split(r"\n{2,}", text)
    cleaned = [" ".join(p.split()) for p in paragraphs]
    return "\n\n".join(p for p in cleaned if p)


def clean_text(text: str) -> str:
    """
    Light cleaning suitable for Ramayana texts:
    - Normalise Unicode to NFC.
    - Remove non-printable control characters (except whitespace).
    - Preserve Sanskrit names, verse numbers, and section headings.
    """
    # Unicode normalisation
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except tab, newline, carriage return
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace smart quotes with straight equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Normalise whitespace
    text = normalise_whitespace(text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """Truncate text to max_chars, breaking at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + suffix


def count_words(text: str) -> int:
    """Return word count of text."""
    return len(text.split())


# ── File / hashing helpers ────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file (for change detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: str) -> str:
    """Return SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Metadata helpers ──────────────────────────────────────────────────────────

def sanitise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    ChromaDB requires metadata values to be str, int, float, or bool.
    Convert everything else to str and drop None values.
    """
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


# ── Token estimation ──────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate (4 chars ≈ 1 token).
    Used when tiktoken is not available or for quick checks.
    """
    return max(1, len(text) // 4)


def try_count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, else fallback to estimate."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return estimate_tokens(text)


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_source_label(metadata: dict[str, Any]) -> str:
    """Build a human-readable source label from chunk metadata."""
    parts: list[str] = []
    if source := metadata.get("source_filename"):
        parts.append(source)
    if chapter := metadata.get("chapter"):
        parts.append(f"Ch. {chapter}")
    if page := metadata.get("page_number"):
        parts.append(f"p. {page}")
    if not parts:
        return metadata.get("chunk_id", "Unknown source")
    return " · ".join(parts)
