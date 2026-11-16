"""Does the system decline when the evidence genuinely isn't there?

The negative set comes from `benchmark.build_negative_set`: real Open RAG
Benchmark questions whose gold document was deliberately left out of the
index. They are well-formed, on-topic, scientific questions — exactly the kind
that tempt a model to answer from parametric knowledge — and the only correct
behaviour is to say so.

A refusal is detected two ways, because an answer can decline without using a
stock phrase: the agent's own route (`not_found`) and the refusal-marker scan
shared with the output guardrail. Anything else counts as a potential
hallucination and is returned in full for inspection.
"""
from __future__ import annotations

import logging

from research_rag.guardrails import is_refusal

logger = logging.getLogger(__name__)


class RefusalEvaluator:
    def __init__(self, rag_system):
        self.rag_system = rag_system

    def evaluate(self, negative_items: list[dict], progress=None) -> dict:
        logger.info(f"Refusal testing on {len(negative_items)} unanswerable question(s)")
        results = []

        for i, item in enumerate(negative_items, start=1):
            question = item["question"]
            result = self.rag_system.ask_full(question)
            declined = result.answer_kind in ("not_found", "out_of_scope") or is_refusal(result.answer)
            results.append({
                "query_id": item.get("query_id"),
                "question": question,
                "answer": result.answer,
                "answer_kind": result.answer_kind,
                "unindexed_doc_id": item.get("unindexed_doc_id"),
                "n_cited_sources": len(result.cited_sources),
                "grounded": result.grounded,
                "correctly_declined": declined,
            })
            if not declined:
                logger.warning(f"[{i}/{len(negative_items)}] possible hallucination: {result.answer[:150]}")
            if progress:
                progress(i, len(negative_items), question)

        n_correct = sum(r["correctly_declined"] for r in results)
        return {
            "refusal_accuracy": round(n_correct / len(results), 4) if results else 0.0,
            "n_questions": len(results),
            "n_correctly_declined": n_correct,
            "hallucinations": [r for r in results if not r["correctly_declined"]],
            "details": results,
        }
