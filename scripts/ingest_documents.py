"""
ingest_documents.py — Command-line script to ingest Ramayana documents into ChromaDB.

Usage:
    python scripts/ingest_documents.py [--rebuild] [--data-dir PATH]

Options:
    --rebuild     Drop the existing index and re-embed all documents.
    --data-dir    Path to the documents directory (default: ./data/documents).
    --verbose     Enable debug logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.document_loader import load_documents
from src.embeddings import get_embedding_model
from src.text_splitter import TextSplitter
from src.utils import setup_logging
from src.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Ramayana documents into ChromaDB."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the existing index and re-embed all documents.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir,
        help=f"Documents directory (default: {settings.data_dir}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("ingest")

    data_dir: Path = args.data_dir
    logger.info("═" * 60)
    logger.info("Ramayana RAG — Document Ingestion")
    logger.info("═" * 60)
    logger.info("Data directory:  %s", data_dir)
    logger.info("Vector store:    %s", settings.vectorstore_dir)
    logger.info("Collection:      %s", settings.chroma_collection_name)
    logger.info("Chunk size:      %d tokens / %d overlap", settings.chunk_size, settings.chunk_overlap)

    if not data_dir.exists():
        logger.error(
            "Data directory '%s' does not exist. "
            "Create it and add Ramayana documents, then run this script again.",
            data_dir,
        )
        return 1

    # Initialise vector store
    vector_store = VectorStore(
        persist_directory=settings.vectorstore_dir,
        collection_name=settings.chroma_collection_name,
    )

    if args.rebuild:
        logger.info("--rebuild flag set. Resetting the collection…")
        vector_store.reset()

    # Determine which files to skip (already indexed)
    existing_hashes: set[str] = set()
    if not args.rebuild:
        existing_hashes = vector_store.get_indexed_file_hashes()
        if existing_hashes:
            logger.info("Skipping %d already-indexed file(s).", len(existing_hashes))

    # Load documents
    t0 = time.perf_counter()
    documents = load_documents(data_dir, skip_hashes=existing_hashes)

    if not documents:
        if vector_store.is_empty():
            logger.warning(
                "No documents found and the index is empty.\n"
                "Add Ramayana source documents to '%s' and run this script again.\n"
                "See data/README.md for recommended sources.",
                data_dir,
            )
            return 0
        else:
            logger.info(
                "No new documents to index. Current index: %d chunks.",
                vector_store.count(),
            )
            return 0

    logger.info("Loaded %d document sections.", len(documents))

    # Split into chunks
    splitter = TextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Produced %d chunks.", len(chunks))

    if not chunks:
        logger.warning("No chunks produced. Check document quality.")
        return 1

    # Embed chunks
    embedding_model = get_embedding_model(settings.embedding_model)
    logger.info("Embedding %d chunks with '%s'…", len(chunks), settings.embedding_model)

    texts = [c.text for c in chunks]
    try:
        embeddings = embedding_model.embed_texts(texts)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return 1

    logger.info("Embedding complete. Adding to ChromaDB…")

    # Store in ChromaDB
    added = vector_store.add_chunks(
        chunk_ids=[c.chunk_id for c in chunks],
        texts=texts,
        embeddings=embeddings,
        metadatas=[c.metadata for c in chunks],
    )

    elapsed = time.perf_counter() - t0
    logger.info("═" * 60)
    logger.info("Ingestion complete!")
    logger.info("  Chunks added/updated: %d", added)
    logger.info("  Total chunks in index: %d", vector_store.count())
    logger.info("  Indexed sources: %s", ", ".join(vector_store.list_sources()))
    logger.info("  Time elapsed: %.1f seconds", elapsed)
    logger.info("═" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
