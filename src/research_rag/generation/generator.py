"""Calls the LLM and assembles the citation-bearing RAGResponse. Citations are
always derived from Chunk metadata here - there is no parameter for a caller
to supply or override sourcing.

The answer is validated against the sources it was given: [S#] markers that
point at nothing are stripped, and each source is flagged `cited` if the
answer actually used it, so the UI can separate "evidence used" from "also
retrieved".
"""
import logging
import time

from research_rag.guardrails import check_citations
from research_rag.llm import LLMClient
from research_rag.models import Chunk, RAGResponse, SourceCitation
from research_rag.models.response import arxiv_url
from .base import BaseGenerator
from .prompt_builder import CitationPromptBuilder

logger = logging.getLogger(__name__)


class AnswerGenerator(BaseGenerator):
    def __init__(self, llm: LLMClient, prompt_builder: CitationPromptBuilder | None = None,
                 corpus_info=None):
        self._llm = llm
        self._prompt_builder = prompt_builder or CitationPromptBuilder()
        self._corpus_info = corpus_info      # optional CorpusInfo: titles + live corpus description

    @property
    def prompt_builder(self) -> CitationPromptBuilder:
        return self._prompt_builder

    def generate(self, question: str, chunks: list[Chunk], *, low_context: bool = False,
                 hardened: bool = False, citation_reminder: bool = False,
                 sub_questions: list[str] | None = None) -> RAGResponse:
        logger.info(f"Generating answer from {len(chunks)} chunk(s) for question: {question!r}")
        corpus_text = self._corpus_info.describe() if self._corpus_info else ""

        t0 = time.monotonic()
        raw_answer = self._llm.chat(
            [
                {"role": "system", "content": self._prompt_builder.build_system_prompt(corpus_text)},
                {"role": "user", "content": self._prompt_builder.build_user_message(
                    question, chunks, low_context=low_context, hardened=hardened,
                    citation_reminder=citation_reminder, sub_questions=sub_questions)},
            ],
            temperature=0.0, purpose="generate",
        )
        logger.info(f"LLM responded in {time.monotonic() - t0:.1f}s ({len(raw_answer)} chars)")

        check = check_citations(raw_answer, n_sources=len(chunks))
        if check.invalid:
            logger.warning(f"Stripped invalid citation marker(s): {check.invalid}")

        sources = [
            SourceCitation(
                paper_id=chunk.paper_id,
                section=chunk.section,
                page_no=chunk.page_no,
                snippet=chunk.text[:200],
                figure_no=chunk.figure_no,
                label=f"S{i}",
                chunk_type=chunk.chunk_type.value,
                score=chunk.score,
                paper_title=chunk.paper_title or (
                    self._corpus_info.title_for(chunk.paper_id) if self._corpus_info else None),
                cited=f"S{i}" in check.used,
                section_id=chunk.section_id,
                url=arxiv_url(chunk.paper_id),
            )
            for i, chunk in enumerate(chunks, start=1)
        ]
        logger.info(f"Attached {len(sources)} source citation(s); {len(check.used)} referenced inline")
        return RAGResponse(answer=check.text, sources=sources, retrieved_chunks=chunks)
