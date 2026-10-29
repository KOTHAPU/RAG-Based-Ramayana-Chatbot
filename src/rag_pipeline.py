"""
rag_pipeline.py — End-to-end RAG orchestration for the Ramayana chatbot.

Wires together:
1. Query rewriting (follow-up resolution)
2. Semantic retrieval
3. Context construction
4. Grounded LLM generation
5. Citation extraction & validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from src.citations import CitationResult, extract_and_validate_citations
from src.config import Settings, settings as default_settings
from src.embeddings import EmbeddingModel, get_embedding_model
from src.generation import GenerationClient, GenerationResponse
from src.prompt_templates import (
    SYSTEM_PROMPT,
    build_context_block,
    build_insufficient_evidence_response,
    build_user_prompt,
)
from src.query_rewriter import QueryRewriter
from src.retriever import Retriever
from src.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    question: str
    rewritten_query: str
    answer: str
    retrieved_chunks: list[SearchResult]
    citation_result: CitationResult
    model: str
    provider: str
    error: str = ""

    @property
    def has_error(self) -> bool:
        return bool(self.error)

    @property
    def has_results(self) -> bool:
        return bool(self.retrieved_chunks)

    @property
    def is_insufficient(self) -> bool:
        return self.citation_result.has_insufficient_evidence or not self.retrieved_chunks


class RAGPipeline:
    """
    Complete RAG pipeline: retrieve → construct context → generate → validate citations.

    Usage::

        pipeline = RAGPipeline.from_settings(settings)
        response = pipeline.answer("Who is Hanuman?", history=[])
    """

    def __init__(
        self,
        retriever: Retriever,
        generation_client: GenerationClient,
        query_rewriter: QueryRewriter,
        cfg: Settings,
    ) -> None:
        self.retriever = retriever
        self.generation_client = generation_client
        self.query_rewriter = query_rewriter
        self.cfg = cfg

    @classmethod
    def from_settings(cls, cfg: Settings) -> "RAGPipeline":
        """Build a RAGPipeline from a Settings object."""
        # Embedding model (cached singleton)
        embedding_model = get_embedding_model(cfg.embedding_model)

        # Vector store
        vector_store = VectorStore(
            persist_directory=cfg.vectorstore_dir,
            collection_name=cfg.chroma_collection_name,
        )

        # Retriever
        retriever = Retriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            similarity_threshold=cfg.retriever_similarity_threshold,
        )

        # Generation client
        gen_client = GenerationClient(
            provider=cfg.llm_provider,
            model=cfg.effective_llm_model,
            api_key=cfg.groq_api_key,
            ollama_base_url=cfg.ollama_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )

        # Query rewriter (uses LLM for best results)
        query_rewriter = QueryRewriter(llm_client=gen_client)

        return cls(
            retriever=retriever,
            generation_client=gen_client,
            query_rewriter=query_rewriter,
            cfg=cfg,
        )

    # ── Non-streaming answer ──────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a question.

        Args:
            question: User's question.
            history: Conversation history as [{"role": ..., "content": ...}].
            top_k: Override for top-k retrieval.
            similarity_threshold: Override for similarity filter.

        Returns:
            RAGResponse with answer, citations, and retrieved chunks.
        """
        history = history or []
        _top_k = top_k if top_k is not None else self.cfg.retriever_top_k

        if not question.strip():
            return self._empty_response(question, "Empty question received.")

        # Step 1: Rewrite follow-up questions
        rewritten = self.query_rewriter.rewrite(question, history)

        # Step 2: Retrieve relevant chunks
        retrieved = self.retriever.retrieve(
            rewritten,
            top_k=_top_k,
            similarity_threshold=similarity_threshold,
        )

        # Step 3: Handle empty retrieval
        if not retrieved:
            logger.info("No relevant chunks retrieved for: '%s'", rewritten)
            answer_text = build_insufficient_evidence_response()
            citation_result = extract_and_validate_citations(
                answer_text + " [INSUFFICIENT_EVIDENCE]", []
            )
            return RAGResponse(
                question=question,
                rewritten_query=rewritten,
                answer=answer_text,
                retrieved_chunks=[],
                citation_result=citation_result,
                model=self.cfg.effective_llm_model,
                provider=self.cfg.llm_provider,
            )

        # Step 4: Build context
        context = build_context_block(retrieved, max_tokens=self.cfg.max_context_tokens)
        user_prompt = build_user_prompt(rewritten, context)

        # Step 5: Generate answer
        try:
            response: GenerationResponse = self.generation_client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            raw_answer = response.content
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return self._empty_response(question, f"Generation failed: {exc}", rewritten, retrieved)

        # Step 6: Extract and validate citations
        citation_result = extract_and_validate_citations(raw_answer, retrieved)

        return RAGResponse(
            question=question,
            rewritten_query=rewritten,
            answer=citation_result.clean_answer,
            retrieved_chunks=retrieved,
            citation_result=citation_result,
            model=response.model,
            provider=response.provider,
        )

    # ── Streaming answer ──────────────────────────────────────────────────────

    def stream_answer(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> tuple[list[SearchResult], Iterator[str]]:
        """
        Execute retrieval immediately, then stream generation tokens.

        Returns:
            (retrieved_chunks, token_iterator) — the chunks are available
            immediately; tokens stream asynchronously.
        """
        history = history or []
        _top_k = top_k if top_k is not None else self.cfg.retriever_top_k

        if not question.strip():
            return [], iter([build_insufficient_evidence_response()])

        rewritten = self.query_rewriter.rewrite(question, history)
        retrieved = self.retriever.retrieve(rewritten, top_k=_top_k, similarity_threshold=similarity_threshold)

        if not retrieved:
            return [], iter([build_insufficient_evidence_response()])

        context = build_context_block(retrieved, max_tokens=self.cfg.max_context_tokens)
        user_prompt = build_user_prompt(rewritten, context)

        token_stream = self.generation_client.stream(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return retrieved, token_stream

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _empty_response(
        self,
        question: str,
        error: str,
        rewritten: str = "",
        retrieved: list[SearchResult] | None = None,
    ) -> RAGResponse:
        retrieved = retrieved or []
        answer = build_insufficient_evidence_response()
        citation_result = extract_and_validate_citations(
            answer + " [INSUFFICIENT_EVIDENCE]", retrieved
        )
        return RAGResponse(
            question=question,
            rewritten_query=rewritten or question,
            answer=answer,
            retrieved_chunks=retrieved,
            citation_result=citation_result,
            model=self.cfg.effective_llm_model,
            provider=self.cfg.llm_provider,
            error=error,
        )

    @property
    def vector_store(self) -> VectorStore:
        return self.retriever.vector_store

    def index_count(self) -> int:
        return self.vector_store.count()

    def is_ready(self) -> bool:
        """Return True if the pipeline has an index with documents."""
        return not self.vector_store.is_empty()
