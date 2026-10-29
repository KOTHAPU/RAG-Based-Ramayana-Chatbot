"""
config.py — Centralised configuration for the Ramayana RAG Chatbot.

All settings are loaded from environment variables (via .env) with safe defaults.
Never hard-code secrets here; read them exclusively from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels above this file (src/config.py → project root)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Load .env from project root (if it exists)
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    """Return stripped env var, st.secrets, or default."""
    val = os.environ.get(key)
    if val is None or not val.strip():
        try:
            import streamlit as st
            if hasattr(st, "secrets") and key in st.secrets:
                val = str(st.secrets[key])
        except Exception:
            pass
    return (val or default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable settings object built from environment variables."""

    # ── Project paths ─────────────────────────────────────────────────────────
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "documents")
    vectorstore_dir: Path = field(
        default_factory=lambda: Path(
            _env("CHROMA_PERSIST_DIRECTORY", str(PROJECT_ROOT / "vectorstore" / "chroma"))
        ).resolve()
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "qwen/qwen3.8-27b")
    )
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3"))
    llm_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1)
    )
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_collection_name: str = field(
        default_factory=lambda: _env("CHROMA_COLLECTION_NAME", "ramayana_documents")
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_top_k: int = field(default_factory=lambda: _env_int("RETRIEVER_TOP_K", 5))
    retriever_similarity_threshold: float = field(
        default_factory=lambda: _env_float("RETRIEVER_SIMILARITY_THRESHOLD", 0.3)
    )

    # ── Generation ────────────────────────────────────────────────────────────
    max_context_tokens: int = field(
        default_factory=lambda: _env_int("MAX_CONTEXT_TOKENS", 4000)
    )

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def is_groq(self) -> bool:
        return self.llm_provider == "groq"

    @property
    def is_ollama(self) -> bool:
        return self.llm_provider == "ollama"

    @property
    def effective_llm_model(self) -> str:
        if self.is_ollama:
            return self.ollama_model
        return self.llm_model


# Singleton settings object — import this everywhere
settings = Settings()
