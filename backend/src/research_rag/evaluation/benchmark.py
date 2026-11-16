"""Builds evaluation sets from the Open RAG Benchmark files.

Three things this module exists to get right:

1. **Only evaluate on queries whose gold document is actually indexed.**
   Scoring a query whose paper was never ingested measures the size of your
   ingestion budget, not the quality of your retriever. `build_eval_set`
   filters to indexed papers and reports how many it dropped.
2. **Stratified sampling.** A random 100-query sample of a 3045-query set is
   ~63% text-only and can easily contain a handful of table queries, so
   multimodal regressions hide inside the noise. Sampling proportionally
   across (type, source) keeps every slice represented and makes repeat runs
   comparable. The seed is fixed so a run is reproducible.
3. **Unanswerable queries for refusal testing.** The benchmark ships 600 hard
   negatives (documents with no queries) but no unanswerable questions. A
   well-defined negative set is built instead by taking queries whose gold
   document was deliberately NOT indexed: the question is real and
   well-formed, the evidence genuinely is not in the index, and the only
   correct behaviour is to decline.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvalSet:
    items: list[dict]
    n_total_queries: int
    n_indexed_papers: int
    n_dropped_unindexed: int

    def summary(self) -> dict:
        by_type: dict[str, int] = defaultdict(int)
        by_source: dict[str, int] = defaultdict(int)
        for item in self.items:
            by_type[item.get("type", "unknown")] += 1
            by_source[item.get("source", "unknown")] += 1
        return {"n_selected": len(self.items), "n_total_queries": self.n_total_queries,
                "n_indexed_papers": self.n_indexed_papers,
                "n_dropped_unindexed": self.n_dropped_unindexed,
                "by_type": dict(by_type), "by_source": dict(by_source)}


def _stratify(items: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional allocation across (type, source), then a seeded shuffle."""
    if n >= len(items):
        return list(items)

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        strata[(item.get("type"), item.get("source"))].append(item)

    rng = random.Random(seed)
    selected: list[dict] = []
    remainders: list[tuple[float, tuple]] = []
    for key, group in strata.items():
        rng.shuffle(group)
        exact = len(group) * n / len(items)
        take = int(exact)
        selected.extend(group[:take])
        remainders.append((exact - take, key))

    # Largest-remainder method so the total lands exactly on n.
    remainders.sort(reverse=True)
    index = {key: int(len(strata[key]) * n / len(items)) for _, key in remainders}
    for _, key in remainders:
        if len(selected) >= n:
            break
        group = strata[key]
        if index[key] < len(group):
            selected.append(group[index[key]])
            index[key] += 1

    rng.shuffle(selected)
    return selected[:n]


def build_eval_set(
    corpus,                       # RAGBenchCorpus
    indexed_paper_ids: list[str] | set[str],
    n: int | None = 100,
    seed: int = 20261003,
    sources: list[str] | None = None,
    types: list[str] | None = None,
) -> EvalSet:
    """Answerable queries: gold document is indexed, so a correct system can answer."""
    queries, qrels, answers = corpus.queries(), corpus.qrels(), corpus.answers()
    indexed = set(indexed_paper_ids)

    candidates: list[dict] = []
    dropped = 0
    for query_id, query in queries.items():
        qrel = qrels.get(query_id)
        if not qrel:
            continue
        if qrel["doc_id"] not in indexed:
            dropped += 1
            continue
        if sources and query.get("source") not in sources:
            continue
        if types and query.get("type") not in types:
            continue
        candidates.append({
            "query_id": query_id,
            "question": query["query"],
            "type": query.get("type"),
            "source": query.get("source"),
            "doc_id": qrel["doc_id"],
            "section_id": qrel.get("section_id"),
            "expected_answer": answers.get(query_id, ""),
        })

    candidates.sort(key=lambda item: item["query_id"])      # deterministic before sampling
    selected = _stratify(candidates, n, seed) if n else candidates
    logger.info(f"Eval set: {len(selected)} query/queries from {len(candidates)} answerable "
                f"({dropped} dropped — gold document not indexed)")
    return EvalSet(selected, len(queries), len(indexed), dropped)


def build_negative_set(
    corpus,
    indexed_paper_ids: list[str] | set[str],
    n: int = 30,
    seed: int = 20261003,
) -> list[dict]:
    """Unanswerable-by-construction queries: real benchmark questions whose gold
    document is NOT in the index. The only correct response is a refusal."""
    queries, qrels = corpus.queries(), corpus.qrels()
    indexed = set(indexed_paper_ids)

    candidates = [
        {"query_id": query_id, "question": query["query"], "type": query.get("type"),
         "source": query.get("source"), "unindexed_doc_id": qrels[query_id]["doc_id"]}
        for query_id, query in queries.items()
        if query_id in qrels and qrels[query_id]["doc_id"] not in indexed
    ]
    candidates.sort(key=lambda item: item["query_id"])
    rng = random.Random(seed)
    rng.shuffle(candidates)
    logger.info(f"Negative set: {min(n, len(candidates))} unanswerable query/queries "
                f"from {len(candidates)} candidates")
    return candidates[:n]
