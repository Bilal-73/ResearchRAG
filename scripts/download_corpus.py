#!/usr/bin/env python
"""Mirror the Open RAG Benchmark dataset locally.

Downloads from https://huggingface.co/datasets/vectara/open_ragbench (arXiv
split) into data/open_ragbench/:

    queries.json  qrels.json  answers.json  pdf_urls.json  corpus/{doc_id}.json

The four metadata files are small (~1.6 MB total). The corpus files are not:
each embeds its figures as base64, so 1000 papers is several GB. By default
only the **judged** papers are fetched — the ~396 documents that queries are
actually scored against — which is everything you need for a benchmark-complete
evaluation. Pass --all for the full 1000 (the extra 600 are hard negatives,
useful only for stress-testing retrieval precision).

Usage:
    python scripts/download_corpus.py                 # metadata + judged papers
    python scripts/download_corpus.py --limit 50      # a quick subset to try things out
    python scripts/download_corpus.py --all           # every paper, including hard negatives
    python scripts/download_corpus.py --metadata-only
"""
import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _bootstrap import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{subdir}"
METADATA_FILES = ["queries.json", "qrels.json", "answers.json", "pdf_urls.json"]


def _fetch(url: str, destination: Path, timeout: float = 120.0) -> int:
    import httpx

    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        size = 0
        with tmp.open("wb") as handle:
            for block in response.iter_bytes(65536):
                handle.write(block)
                size += len(block)
        tmp.replace(destination)        # a killed download can't leave a half-written .json behind
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Download all 1000 papers, not only the judged ones.")
    parser.add_argument("--limit", type=int, default=None, help="Download at most N corpus files.")
    parser.add_argument("--metadata-only", action="store_true", help="Only queries/qrels/answers/pdf_urls.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (default 8).")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    args = parser.parse_args()

    settings = get_settings()
    base = HF_BASE.format(repo=settings.hf_dataset_repo, subdir=settings.hf_dataset_subdir)
    out_dir = Path(settings.open_ragbench_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    logger.info(f"Source: {settings.hf_dataset_repo} ({settings.hf_dataset_subdir})")
    logger.info(f"Target: {out_dir}")

    for name in METADATA_FILES:
        destination = out_dir / name
        if destination.exists() and not args.force:
            logger.info(f"  {name}: already present, skipping")
            continue
        size = _fetch(f"{base}/{name}", destination)
        logger.info(f"  {name}: {size / 1024:.0f} KB")

    if args.metadata_only:
        logger.info("Metadata only — done.")
        return

    qrels = json.loads((out_dir / "qrels.json").read_text(encoding="utf-8"))
    pdf_urls = json.loads((out_dir / "pdf_urls.json").read_text(encoding="utf-8"))
    judged = sorted({entry["doc_id"] for entry in qrels.values()})
    paper_ids = sorted(pdf_urls) if args.all else judged
    if args.limit:
        paper_ids = paper_ids[:args.limit]

    logger.info(f"{len(judged)} judged paper(s) of {len(pdf_urls)} total; downloading {len(paper_ids)}")
    corpus_dir = out_dir / "corpus"
    todo = [pid for pid in paper_ids if args.force or not (corpus_dir / f"{pid}.json").exists()]
    logger.info(f"{len(paper_ids) - len(todo)} already present; fetching {len(todo)}")
    if not todo:
        return

    t0 = time.monotonic()
    done = failed = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, f"{base}/corpus/{pid}.json", corpus_dir / f"{pid}.json"): pid
                   for pid in todo}
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                total_bytes += future.result()
                done += 1
            except Exception as error:  # noqa: BLE001 — report and keep going
                failed += 1
                logger.error(f"  {paper_id}: {error}")
            if (done + failed) % 25 == 0 or (done + failed) == len(todo):
                elapsed = time.monotonic() - t0
                print(f"  {done + failed}/{len(todo)} files · {total_bytes / 1e6:.0f} MB · {elapsed:.0f}s",
                      file=sys.stderr)

    logger.info(f"Downloaded {done} file(s), {failed} failure(s), {total_bytes / 1e6:.0f} MB "
                f"in {time.monotonic() - t0:.0f}s")
    print(f"\nCorpus ready at {out_dir}. Next: python scripts/run_ingestion.py")


if __name__ == "__main__":
    main()
