"""Retrieval metrics against Open RAG Benchmark qrels.

The benchmark judges relevance at **(doc_id, section_id)** granularity, and
this system indexes several chunks per section, so a retrieved chunk counts as
a hit when its paper_id AND section_id match the qrel. Two strictnesses are
reported because they answer different questions:

* ``doc``      — did we find the right *paper*? (loose; what a paper-level
                 benchmark would score)
* ``section``  — did we find the right *section*? (strict; what the benchmark
                 actually judges, and what the generator needs to be correct)

Metrics are standard: Recall@k (binary, since each query has one judged
section), MRR@k and NDCG@k. They are also broken out by the query's `type`
(abstractive/extractive) and `source` (text / text-image / text-table /
text-table-image), which is where a multimodal pipeline earns or loses its
keep — a text-only indexer scores near zero on table- and image-sourced
queries, and the breakdown makes that visible instead of hiding it in an
average.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

from research_rag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


def _dcg_at_1_hit(rank: int | None) -> float:
    """NDCG with a single relevant item: IDCG = 1, so NDCG = 1/log2(rank+1)."""
    return 1.0 / math.log2(rank + 1) if rank else 0.0


class RetrievalEvaluator:
    """`eval_items` come from `evaluation.benchmark.build_eval_set`:
    {"query_id", "question", "doc_id", "section_id", "type", "source"}."""

    def __init__(self, retriever: BaseRetriever, top_k: int = 10, reranker=None, rerank_top_k: int | None = None):
        self.retriever = retriever
        self.top_k = top_k
        self.reranker = reranker              # optional: measures the reranked list instead of raw retrieval
        self.rerank_top_k = rerank_top_k or top_k

    def _ranks(self, chunks, doc_id: str, section_id) -> tuple[int | None, int | None]:
        doc_rank = section_rank = None
        for rank, chunk in enumerate(chunks, start=1):
            if chunk.paper_id != doc_id:
                continue
            if doc_rank is None:
                doc_rank = rank
            if section_id is not None and chunk.section_id == section_id and section_rank is None:
                section_rank = rank
        return doc_rank, section_rank

    def evaluate(self, eval_items: list[dict], progress=None) -> dict:
        logger.info(f"Evaluating retrieval on {len(eval_items)} query/queries, top_k={self.top_k}")
        rows: list[dict] = []

        for i, item in enumerate(eval_items, start=1):
            question = item["question"]
            chunks = self.retriever.retrieve(question, top_k=self.top_k)
            if self.reranker is not None:
                chunks = self.reranker.rerank(question, chunks, top_k=self.rerank_top_k)

            doc_rank, section_rank = self._ranks(chunks, item["doc_id"], item.get("section_id"))
            rows.append({
                "query_id": item.get("query_id"), "question": question,
                "doc_id": item["doc_id"], "section_id": item.get("section_id"),
                "type": item.get("type", "unknown"), "source": item.get("source", "unknown"),
                "doc_rank": doc_rank, "section_rank": section_rank,
                "retrieved": [{"paper_id": c.paper_id, "section_id": c.section_id,
                               "chunk_type": c.chunk_type.value} for c in chunks],
            })
            if progress:
                progress(i, len(eval_items), question)

        return {"top_k": self.top_k, "n_questions": len(rows),
                "overall": _aggregate(rows),
                "by_type": _group(rows, "type"),
                "by_source": _group(rows, "source"),
                "details": rows}


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out: dict[str, float] = {}
    for level, key in (("doc", "doc_rank"), ("section", "section_rank")):
        ranks = [row[key] for row in rows]
        out[f"recall@k_{level}"] = sum(1 for r in ranks if r) / len(ranks)
        out[f"mrr_{level}"] = sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks)
        out[f"ndcg@k_{level}"] = sum(_dcg_at_1_hit(r) for r in ranks) / len(ranks)
    out["n"] = len(rows)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def _group(rows: list[dict], field: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    return {name: _aggregate(items) for name, items in sorted(grouped.items())}
