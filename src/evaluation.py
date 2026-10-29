"""
evaluation.py — RAG evaluation metrics for the Ramayana chatbot.

Computes retrieval-level metrics (Recall@k, Precision@k, MRR, Hit Rate)
and citation validity rate.  Generation quality is deliberately kept
separate and optional (requires external LLM judge).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Metric data classes ───────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Retrieval-level evaluation results for a single query."""

    query: str
    expected_sources: list[str]  # Source filenames or keywords expected
    retrieved_sources: list[str]  # Sources actually retrieved
    top_k: int

    @property
    def hits(self) -> list[str]:
        """Expected sources that were retrieved."""
        expected_lower = {s.lower() for s in self.expected_sources}
        return [s for s in self.retrieved_sources if s.lower() in expected_lower]

    @property
    def recall_at_k(self) -> float:
        """Recall@k: fraction of expected sources that were retrieved."""
        if not self.expected_sources:
            return 1.0
        return len(self.hits) / len(self.expected_sources)

    @property
    def precision_at_k(self) -> float:
        """Precision@k: fraction of retrieved that were relevant."""
        if not self.retrieved_sources:
            return 0.0
        return len(self.hits) / len(self.retrieved_sources)

    @property
    def hit_rate(self) -> float:
        """1 if at least one expected source was retrieved, else 0."""
        return 1.0 if self.hits else 0.0

    @property
    def reciprocal_rank(self) -> float:
        """MRR contribution: 1/rank of the first relevant result, or 0."""
        expected_lower = {s.lower() for s in self.expected_sources}
        for rank, source in enumerate(self.retrieved_sources, start=1):
            if source.lower() in expected_lower:
                return 1.0 / rank
        return 0.0


@dataclass
class EvaluationReport:
    """Aggregated evaluation report over all test cases."""

    results: list[RetrievalMetrics] = field(default_factory=list)
    citation_validity_rates: list[float] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall_at_k for r in self.results) / len(self.results)

    @property
    def mean_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.precision_at_k for r in self.results) / len(self.results)

    @property
    def mean_hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.hit_rate for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank."""
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def mean_citation_validity(self) -> float:
        if not self.citation_validity_rates:
            return 1.0
        return sum(self.citation_validity_rates) / len(self.citation_validity_rates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_queries": len(self.results),
            "mean_recall_at_k": round(self.mean_recall, 4),
            "mean_precision_at_k": round(self.mean_precision, 4),
            "mean_hit_rate": round(self.mean_hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "mean_citation_validity": round(self.mean_citation_validity, 4),
            "per_query": [
                {
                    "query": r.query,
                    "recall": round(r.recall_at_k, 4),
                    "precision": round(r.precision_at_k, 4),
                    "hit_rate": r.hit_rate,
                    "rr": round(r.reciprocal_rank, 4),
                }
                for r in self.results
            ],
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("RAG EVALUATION REPORT")
        print("=" * 60)
        print(f"Queries evaluated:      {len(self.results)}")
        print(f"Mean Recall@k:          {self.mean_recall:.4f}")
        print(f"Mean Precision@k:       {self.mean_precision:.4f}")
        print(f"Mean Hit Rate:          {self.mean_hit_rate:.4f}")
        print(f"MRR:                    {self.mrr:.4f}")
        print(f"Citation Validity:      {self.mean_citation_validity:.4f}")
        print("=" * 60 + "\n")


# ── Evaluation test dataset ───────────────────────────────────────────────────

# These are evaluation question templates.  The expected_sources field lists
# keywords / filenames that a correctly indexed Ramayana document should contain.
# These are NOT fabricated answers — they describe what relevant documents should
# contain, not what the LLM should say.

EVALUATION_DATASET: list[dict[str, Any]] = [
    {
        "query": "Who is Hanuman and what is his role in the Ramayana?",
        "expected_keywords": ["hanuman", "vayu", "monkey", "lanka", "sita"],
        "notes": "Core character query — should retrieve character introduction passages.",
    },
    {
        "query": "Why was Rama exiled from Ayodhya?",
        "expected_keywords": ["ayodhya", "exile", "kaikeyi", "dasharatha", "fourteen", "forest"],
        "notes": "Key story event — Kaikeyi's boon leading to 14-year exile.",
    },
    {
        "query": "How did Hanuman find Sita in Lanka?",
        "expected_keywords": ["lanka", "sita", "hanuman", "ashoka", "garden"],
        "notes": "Sundarakanda event — Hanuman's search in Lanka.",
    },
    {
        "query": "What is the significance of the construction of the bridge to Lanka?",
        "expected_keywords": ["bridge", "lanka", "nala", "ocean", "vanaras"],
        "notes": "Yuddha Kanda event — construction of Rama Setu.",
    },
    {
        "query": "Who is Ravana and why did he abduct Sita?",
        "expected_keywords": ["ravana", "sita", "abduct", "lanka", "demon"],
        "notes": "Antagonist characterisation and motivation.",
    },
    {
        "query": "What role did Vibhishana play in the war?",
        "expected_keywords": ["vibhishana", "ravana", "lanka", "defected", "advice"],
        "notes": "Vibhishana's defection and advice to Rama.",
    },
    {
        "query": "Describe the relationship between Rama and Lakshmana.",
        "expected_keywords": ["lakshmana", "rama", "brother", "exile", "loyal"],
        "notes": "Character relationship — sibling loyalty.",
    },
    {
        "query": "What happened during the swayamvara of Sita?",
        "expected_keywords": ["sita", "swayamvara", "bow", "janaka", "mithila"],
        "notes": "Bala Kanda event — Sita's marriage to Rama.",
    },
]


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates retrieval quality against the test dataset.

    Does NOT require a running LLM — only the retriever and vector store.
    """

    def __init__(self, retriever) -> None:
        self.retriever = retriever

    def evaluate(
        self,
        dataset: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> EvaluationReport:
        """
        Run evaluation over the test dataset.

        Args:
            dataset: List of test cases. Uses EVALUATION_DATASET if None.
            top_k: Number of chunks to retrieve per query.

        Returns:
            EvaluationReport with aggregated metrics.
        """
        dataset = dataset or EVALUATION_DATASET
        report = EvaluationReport()

        logger.info("Running retrieval evaluation over %d queries…", len(dataset))

        for case in dataset:
            query = case["query"]
            expected_keywords = [kw.lower() for kw in case.get("expected_keywords", [])]

            try:
                results = self.retriever.retrieve(query, top_k=top_k)
                # Extract keywords present in retrieved texts
                retrieved_keywords: list[str] = []
                for result in results:
                    text_lower = result.text.lower()
                    for kw in expected_keywords:
                        if kw in text_lower and kw not in retrieved_keywords:
                            retrieved_keywords.append(kw)

                metric = RetrievalMetrics(
                    query=query,
                    expected_sources=expected_keywords,
                    retrieved_sources=retrieved_keywords,
                    top_k=top_k,
                )
                report.results.append(metric)
                logger.debug(
                    "Query: '%s…' | Recall@%d: %.2f | Hit: %s",
                    query[:40],
                    top_k,
                    metric.recall_at_k,
                    bool(metric.hits),
                )
            except Exception as exc:
                logger.error("Evaluation failed for query '%s': %s", query, exc)

        return report

    def save_report(self, report: EvaluationReport, output_path: Path) -> None:
        """Save evaluation report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Evaluation report saved to '%s'.", output_path)
