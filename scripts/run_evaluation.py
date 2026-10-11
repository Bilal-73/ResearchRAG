#!/usr/bin/env python
"""Evaluate the system against the Open RAG Benchmark.

Three stages, each independently useful:

  1. Retrieval metrics — Recall@k / MRR / NDCG at both document and *section*
     granularity (the benchmark judges sections), broken out by query type
     (abstractive/extractive) and modality (text / table / image). Much cheaper
     than a full run, though not free: query rewriting and HyDE still cost 2 LLM
     calls per query unless you disable them for the run.
  2. End-to-end answer quality — RAGAS by default: faithfulness,
     answer_relevancy, context_precision and context_recall, with the judge
     models wired to this project's Azure deployments. `--judge builtin` swaps
     in the dependency-free judge, which adds correctness against the
     benchmark's reference answer and citation quality.
  3. Refusal testing — real benchmark questions whose gold document was
     deliberately left out of the index. The only correct answer is "not in
     the retrieved context"; anything else is a hallucination.

Only queries whose gold document is actually indexed are scored, and the
sample is stratified across (type, source) with a fixed seed so runs are
comparable. Results are written to data/eval/results_<timestamp>.json.

Usage:
    python scripts/run_evaluation.py                     # 100 queries, all three stages
    python scripts/run_evaluation.py -n 25               # quick pass
    python scripts/run_evaluation.py --retrieval-only    # skip generation + judging
    python scripts/run_evaluation.py --sources text-table text-table-image
    python scripts/run_evaluation.py --no-rerank         # measure raw retrieval
    python scripts/run_evaluation.py --judge builtin     # dependency-free judge instead of RAGAS
"""
import argparse
import json
import logging
import sys
from datetime import datetime

from _bootstrap import PROJECT_ROOT, build_agent, build_query_components, get_settings

from research_rag.evaluation import (
    LLMJudgeEvaluator, RefusalEvaluator, RetrievalEvaluator, build_eval_set, build_negative_set,
)
from research_rag.ingestion import RAGBenchCorpus
from research_rag.rag_system import RAGSystem

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "data" / "eval"


def _resolve_corpus_dir(settings):
    from pathlib import Path

    path = Path(settings.open_ragbench_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _progress(label: str):
    def report(done: int, total: int, text: str) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {label}: {done}/{total}", file=sys.stderr)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n-queries", type=int, default=100, help="Answerable queries to score (0 = all).")
    parser.add_argument("--n-negative", type=int, default=25, help="Unanswerable queries for refusal testing.")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval depth for the metrics pass.")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Filter by modality: text text-image text-table text-table-image")
    parser.add_argument("--types", nargs="*", default=None, help="Filter by abstractive / extractive.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Stage 1 only: skip generation and judging. Still pays query "
                             "rewrite + HyDE unless those are disabled.")
    parser.add_argument("--no-rerank", action="store_true", help="Score retrieval before reranking.")
    parser.add_argument("--judge", choices=["ragas", "builtin"], default="ragas",
                        help="Answer-quality scorer. 'ragas' (default) gives faithfulness, "
                             "answer_relevancy, context_precision and context_recall. 'builtin' is the "
                             "dependency-free judge (adds correctness vs the reference answer and "
                             "citation quality).")
    parser.add_argument("--out", default=None, help="Results file (default: data/eval/results_<ts>.json).")
    args = parser.parse_args()

    settings = get_settings()
    components = build_query_components(settings)
    corpus = RAGBenchCorpus(_resolve_corpus_dir(settings))
    corpus_info = components["corpus"]
    indexed = corpus_info.ingested_ids(force=True)
    if not indexed:
        logger.error("Nothing is indexed. Run: python scripts/run_ingestion.py")
        sys.exit(1)

    eval_set = build_eval_set(corpus, indexed, n=args.n_queries or None, seed=args.seed,
                              sources=args.sources, types=args.types)
    print("\nEval set:\n" + json.dumps(eval_set.summary(), indent=2))
    if not eval_set.items:
        logger.error("No answerable queries matched. Index more papers or loosen the filters.")
        sys.exit(1)

    results = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "n_queries": len(eval_set.items), "seed": args.seed, "sources": args.sources, "types": args.types,
            "top_k": args.top_k or settings.retrieval_top_k, "rerank": not args.no_rerank,
            "rerank_top_k": settings.rerank_top_k,
            "enable_query_rewrite": settings.enable_query_rewrite, "enable_hyde": settings.enable_hyde,
            "enable_planner": settings.enable_planner,
            "enable_context_grading": settings.enable_context_grading,
            "chunk_max_tokens": settings.chunk_max_tokens, "caption_figures": settings.caption_figures,
        },
        "eval_set": eval_set.summary(),
    }

    print("\n--- 1/3: retrieval metrics ---", file=sys.stderr)
    retrieval = RetrievalEvaluator(
        components["retriever"], top_k=args.top_k or settings.retrieval_top_k,
        reranker=None if args.no_rerank else components["reranker"],
        rerank_top_k=settings.rerank_top_k,
    ).evaluate(eval_set.items, progress=_progress("retrieval"))
    results["retrieval"] = {k: v for k, v in retrieval.items() if k != "details"}
    results["retrieval_details"] = retrieval["details"]
    print("\nRetrieval:\n" + json.dumps(results["retrieval"], indent=2))

    if not args.retrieval_only:
        agent, _registry, _limiter, llm, _corpus, _cache = build_agent(settings)
        rag_system = RAGSystem(agent, components["retriever"], components["reranker"])

        print("\n--- 2/3: end-to-end answer quality ---", file=sys.stderr)
        results["config"]["judge"] = args.judge
        if args.judge == "ragas":
            from research_rag.evaluation import RagasEvaluator

            try:
                results["answer_quality"] = RagasEvaluator(rag_system, settings).evaluate(
                    eval_set.items, progress=_progress("ragas"))
            except Exception as error:  # noqa: BLE001 — RAGAS's dependency chain is fragile
                logger.warning(f"RAGAS failed ({error}) — falling back to the built-in judge")
                results["config"]["judge"] = "builtin (ragas failed)"
                results["answer_quality"] = LLMJudgeEvaluator(rag_system, llm).evaluate(
                    eval_set.items, progress=_progress("judge"))
        else:
            results["answer_quality"] = LLMJudgeEvaluator(rag_system, llm).evaluate(
                eval_set.items, progress=_progress("judge"))
        print("\nAnswer quality:\n" + json.dumps(
            {k: v for k, v in results["answer_quality"].items() if k != "details"}, indent=2))

        print("\n--- 3/3: refusal testing ---", file=sys.stderr)
        negatives = build_negative_set(corpus, indexed, n=args.n_negative, seed=args.seed)
        if negatives:
            refusal = RefusalEvaluator(rag_system).evaluate(negatives, progress=_progress("refusal"))
            results["refusal"] = {k: v for k, v in refusal.items() if k != "details"}
            results["refusal_details"] = refusal["details"]
            print("\nRefusal:\n" + json.dumps(
                {k: v for k, v in results["refusal"].items() if k != "hallucinations"}, indent=2))
            for item in refusal["hallucinations"]:
                print(f"\n  [NOT DECLINED] {item['question']}\n    -> {item['answer'][:200]}")
        else:
            logger.info("Every paper is indexed — no unanswerable queries available for refusal testing.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (RESULTS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json") if not args.out else \
        (PROJECT_ROOT / args.out)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
