"""Top-level facade used by evaluation and one-shot CLI queries.

It wraps the LangGraph agent rather than re-implementing retrieve → rerank →
generate, so **evaluation measures exactly what the API serves** — the same
routing, the same guardrails, the same corrective-retrieval loop. A separate
"simple" path would score a pipeline no user ever hits.

Each `ask()` runs on a throwaway thread id, so evaluation questions never leak
conversational memory into one another.
"""
from __future__ import annotations

import logging
import uuid

from langchain_core.runnables import Runnable, RunnableLambda

from research_rag.models import Chunk, RAGResponse, SourceCitation

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, agent, retriever=None, reranker=None):
        self.agent = agent
        self.retriever = retriever        # exposed so retrieval-only metrics can skip generation
        self.reranker = reranker

    def ask(self, question: str, *, options: dict | None = None) -> RAGResponse:
        result = self.agent.chat(question, f"eval-{uuid.uuid4().hex[:12]}", options)
        return RAGResponse(
            answer=result.answer,
            sources=list(result.sources),
            retrieved_chunks=list(result.chunks),
        )

    def ask_full(self, question: str, *, options: dict | None = None):
        """The complete AgentResult (intent, trace, guardrail events, groundedness)."""
        return self.agent.chat(question, f"eval-{uuid.uuid4().hex[:12]}", options)

    def as_runnable(self) -> Runnable:
        """The whole system as one LangChain Runnable: question (str) -> RAGResponse, so it
        composes with other LCEL chains, batch() and callback/tracing configs."""
        return RunnableLambda(lambda question: self.ask(question), name="research_rag")


__all__ = ["RAGSystem", "RAGResponse", "SourceCitation", "Chunk"]
