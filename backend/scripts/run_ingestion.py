#!/usr/bin/env python
"""Index the Open RAG Benchmark corpus into Qdrant.

Resumable: progress is recorded in the manifest after each paper, so a run that
is interrupted (or rate-limited) can simply be started again — papers already
indexed are skipped unless --force is passed.

Usage:
    python scripts/run_ingestion.py                      # every downloaded paper
    python scripts/run_ingestion.py --judged-only        # only papers the benchmark scores against
    python scripts/run_ingestion.py --limit 25           # a quick subset
    python scripts/run_ingestion.py --paper 2410.14077v2 # one paper
    python scripts/run_ingestion.py --force              # re-index everything
    python scripts/run_ingestion.py --dry-run            # chunk only; no embedding, no Qdrant writes
"""
import argparse
import json
import logging
import sys

from _bootstrap import build_ingestion, get_settings

logger = logging.getLogger(__name__)


def _progress(done: int, total: int, label: str) -> None:
    print(f"  [{done}/{total}] {label}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", default=None, help="Index only this paper id.")
    parser.add_argument("--judged-only", action="store_true",
                        help="Only papers referenced by the benchmark's qrels.")
    parser.add_argument("--limit", type=int, default=None, help="Index at most N papers.")
    parser.add_argument("--force", action="store_true", help="Re-index papers already in the manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chunk and report statistics without embedding or writing to Qdrant.")
    args = parser.parse_args()

    settings = get_settings()
    pipeline, corpus, components = build_ingestion(settings)

    available = corpus.available_ids()
    if not available:
        logger.error(f"No corpus files in {corpus.corpus_dir}. Run: python scripts/download_corpus.py")
        sys.exit(1)

    if args.paper:
        paper_ids = [args.paper]
    elif args.judged_only:
        paper_ids = [pid for pid in corpus.judged_ids() if pid in set(available)]
    else:
        paper_ids = available
    if args.limit:
        paper_ids = paper_ids[:args.limit]

    if args.dry_run:
        totals: dict[str, int] = {}
        chars = 0
        for paper_id in paper_ids:
            chunks = pipeline.chunker.chunk_document(corpus.document(paper_id))
            for chunk in chunks:
                totals[chunk.chunk_type.value] = totals.get(chunk.chunk_type.value, 0) + 1
                chars += len(chunk.text)
        n = sum(totals.values())
        print(json.dumps({"dry_run": True, "papers": len(paper_ids), "chunks": n, "by_type": totals,
                          "avg_chunk_chars": round(chars / n) if n else 0}, indent=2))
        return

    logger.info(f"Indexing {len(paper_ids)} paper(s) into '{settings.qdrant_collection_name}'")
    dense_dim = len(components["embeddings"].embed_query("dimension probe"))
    components["vector_store"].create_collection(dense_dim=dense_dim)
    logger.info(f"Qdrant collection ready (dense dim={dense_dim})")

    report = pipeline.ingest(paper_ids, force=args.force, progress=_progress)
    print("\n" + json.dumps(report.to_dict(), indent=2))
    if report.failures:
        print(f"\n{len(report.failures)} paper(s) failed — re-run to retry just those.", file=sys.stderr)


if __name__ == "__main__":
    main()
