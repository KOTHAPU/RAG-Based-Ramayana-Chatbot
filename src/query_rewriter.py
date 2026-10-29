"""
query_rewriter.py — Rewrite follow-up questions into self-contained retrieval queries.

Uses a lightweight LLM call to resolve pronouns and references from conversation history.
Falls back gracefully to the original question if the LLM is unavailable.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# If a question references these, it's almost certainly a follow-up
_FOLLOWUP_INDICATORS = re.compile(
    r"\b(he|she|they|it|him|her|them|his|its|their|"
    r"this|that|these|those|who|what|where|when|why|how)\b",
    re.IGNORECASE,
)

_PRONOUNS = re.compile(
    r"\b(he|she|they|it|him|her|them|his|its|their)\b",
    re.IGNORECASE,
)

REWRITE_SYSTEM_PROMPT = """You are a question clarification assistant for a Ramayana knowledge base.
Given a conversation history and a follow-up question, rewrite the question into a completely 
self-contained, standalone query suitable for semantic document retrieval.

Rules:
- Replace all pronouns with the specific names/entities they refer to.
- Include relevant context from the conversation history.
- Keep the rewritten query concise (one sentence if possible).
- Do NOT answer the question — only rewrite it.
- If the question is already self-contained, return it unchanged.
- Output ONLY the rewritten question, nothing else.
"""


class QueryRewriter:
    """
    Rewrites conversational follow-up questions into standalone retrieval queries.

    Works best with a Groq/Ollama LLM. Falls back to heuristic rewriting when
    the LLM is unavailable.
    """

    def __init__(self, llm_client=None) -> None:
        """
        Args:
            llm_client: Optional LLM client (GenerationClient). If None, uses
                        heuristic fallback only.
        """
        self.llm_client = llm_client

    def needs_rewriting(self, question: str, history: list[dict]) -> bool:
        """Return True if the question likely needs context from history."""
        if not history:
            return False
        if len(question.split()) < 8 and _FOLLOWUP_INDICATORS.search(question):
            return True
        if _PRONOUNS.search(question):
            return True
        return False

    def rewrite(self, question: str, history: list[dict]) -> str:
        """
        Rewrite a follow-up question into a standalone query.

        Args:
            question: The user's latest question.
            history: List of {"role": "user"|"assistant", "content": str} dicts.

        Returns:
            Rewritten question (or original if no rewriting needed / possible).
        """
        if not self.needs_rewriting(question, history):
            logger.debug("Query does not need rewriting: '%s'", question)
            return question

        # Try LLM-based rewriting first
        if self.llm_client is not None:
            try:
                rewritten = self._llm_rewrite(question, history)
                if rewritten and rewritten != question:
                    logger.info("Rewritten query: '%s' → '%s'", question, rewritten)
                    return rewritten
            except Exception as exc:
                logger.warning("LLM query rewriting failed: %s. Using heuristic.", exc)

        # Heuristic fallback
        return self._heuristic_rewrite(question, history)

    def _llm_rewrite(self, question: str, history: list[dict]) -> str:
        """Use the LLM to rewrite the question."""
        # Build a compact history representation (last 3 exchanges)
        recent = history[-6:] if len(history) > 6 else history
        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent
        )

        prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {question}\n\n"
            f"Rewritten standalone query:"
        )

        response = self.llm_client.generate(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=150,
            temperature=0.0,
        )

        rewritten = response.content.strip()
        # Strip common prefixes the model might add
        for prefix in ["Rewritten standalone query:", "Standalone query:", "Query:"]:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        return rewritten or question

    def _heuristic_rewrite(self, question: str, history: list[dict]) -> str:
        """
        Simple heuristic: prepend the last mentioned entity name to the question.
        Very basic but avoids total failure when LLM is unavailable.
        """
        if not history:
            return question

        # Look for proper nouns (capitalised words) in recent messages
        all_text = " ".join(msg["content"] for msg in history[-4:])
        proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", all_text)

        # Filter out common non-entities
        stop_words = {"The", "This", "That", "When", "Where", "What", "How", "Who", "Why"}
        entities = [n for n in proper_nouns if n not in stop_words]

        if entities:
            # Use the most recent entity
            entity = entities[-1]
            if entity.lower() not in question.lower():
                rewritten = f"{question.rstrip('?')} (regarding {entity})?"
                logger.debug("Heuristic rewrite: '%s' → '%s'", question, rewritten)
                return rewritten

        return question
