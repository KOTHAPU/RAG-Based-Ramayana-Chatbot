"""
vector_store.py — ChromaDB persistent vector store operations.

Provides a clean interface for:
- Creating / opening a persistent ChromaDB collection.
- Adding chunks with embeddings and metadata.
- Querying by vector similarity.
- Collection statistics and management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result from a vector similarity search."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float  # Lower is more similar (cosine distance)
    rank: int

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity score [0, 1]."""
        return max(0.0, 1.0 - self.distance)

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, score={self.similarity_score:.3f}, "
            f"source={self.metadata.get('source_filename', '?')!r})"
        )


class VectorStore:
    """
    Persistent ChromaDB vector store for Ramayana document chunks.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "ramayana_documents",
        embedding_function=None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_function = embedding_function
        self._initialise()

    def _initialise(self) -> None:
        """Open or create the persistent ChromaDB collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            # Get or create the collection (no embedding function; we pass embeddings explicitly)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' opened at '%s' (%d chunks).",
                self.collection_name,
                self.persist_directory,
                self.count(),
            )
        except ImportError:
            raise RuntimeError("chromadb is not installed. Run: pip install chromadb")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise ChromaDB: {exc}") from exc

    # ── Write operations ──────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert chunks into the collection in batches.
        Returns the number of chunks added/updated.
        """
        if not chunk_ids:
            return 0

        total = len(chunk_ids)
        added = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            try:
                self._collection.upsert(
                    ids=chunk_ids[start:end],
                    documents=texts[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )
                added += end - start
                logger.debug("Upserted batch %d–%d / %d.", start + 1, end, total)
            except Exception as exc:
                logger.error("Failed to upsert batch %d–%d: %s", start + 1, end, exc)

        logger.info("Added/updated %d chunks in collection '%s'.", added, self.collection_name)
        return added

    def delete_by_source(self, source_filename: str) -> None:
        """Delete all chunks from a specific source file."""
        try:
            self._collection.delete(where={"source_filename": source_filename})
            logger.info("Deleted chunks for source '%s'.", source_filename)
        except Exception as exc:
            logger.warning("Could not delete chunks for '%s': %s", source_filename, exc)

    def reset(self) -> None:
        """Drop and recreate the collection (full rebuild)."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection '%s' reset.", self.collection_name)
        except Exception as exc:
            logger.error("Failed to reset collection: %s", exc)

    # ── Read operations ───────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Query the collection by vector similarity.

        Returns SearchResult list sorted by ascending distance (most similar first).
        """
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, self.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        search_results: list[SearchResult] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for rank, (doc, meta, dist, cid) in enumerate(
            zip(documents, metadatas, distances, ids), start=1
        ):
            search_results.append(
                SearchResult(
                    chunk_id=cid,
                    text=doc or "",
                    metadata=meta or {},
                    distance=float(dist),
                    rank=rank,
                )
            )

        return search_results

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        """Return a deduplicated list of indexed source filenames."""
        try:
            result = self._collection.get(include=["metadatas"])
            filenames = {
                m.get("source_filename", "unknown")
                for m in (result.get("metadatas") or [])
            }
            return sorted(filenames)
        except Exception as exc:
            logger.error("Could not list sources: %s", exc)
            return []

    def get_indexed_file_hashes(self) -> set[str]:
        """Return all file hashes already stored in the collection."""
        try:
            result = self._collection.get(include=["metadatas"])
            return {
                m["file_hash"]
                for m in (result.get("metadatas") or [])
                if "file_hash" in m
            }
        except Exception as exc:
            logger.error("Could not retrieve file hashes: %s", exc)
            return set()

    def is_empty(self) -> bool:
        return self.count() == 0
