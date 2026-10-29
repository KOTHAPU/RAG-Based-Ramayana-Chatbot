"""
evaluate_rag.py — Run the RAG evaluation suite.

Requires a populated ChromaDB index. Evaluation runs retrieval-only metrics
(no LLM call required) plus optional citation validity checks.

Usage:
    python scripts/evaluate_rag.py [--top-k N] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.embeddings import get_embedding_model
from src.evaluation import EVALUATION_DATASET, RAGEvaluator
from src.retriever import Retriever
from src.utils import setup_logging
from src.vector_store import VectorStore
import logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("--top-k", type=int, default=settings.retriever_top_k)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output file for the JSON report.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("evaluate")

    logger.info("═" * 60)
    logger.info("Ramayana RAG — Evaluation Suite")
    logger.info("═" * 60)

    # Check index
    vector_store = VectorStore(
        persist_directory=settings.vectorstore_dir,
        collection_name=settings.chroma_collection_name,
    )
    if vector_store.is_empty():
        logger.error(
            "Vector store is empty. Ingest documents first:\n"
            "  python scripts/ingest_documents.py"
        )
        return 1

    logger.info("Index contains %d chunks.", vector_store.count())
    logger.info("Evaluation queries: %d", len(EVALUATION_DATASET))

    # Build retriever
    embedding_model = get_embedding_model(settings.embedding_model)
    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        similarity_threshold=0.0,  # No threshold for evaluation
    )

    # Run evaluation
    evaluator = RAGEvaluator(retriever=retriever)
    report = evaluator.evaluate(top_k=args.top_k)
    report.print_summary()

    # Save report
    evaluator.save_report(report, args.output)
    logger.info("Full report saved to '%s'.", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
