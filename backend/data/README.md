# data/

Everything here is generated — nothing in this folder is hand-maintained input.

- **`open_ragbench/`** — the mirrored
  [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) arXiv split, written
  by `python scripts/download_corpus.py`. Not committed to git (several GB with all papers).
  - `queries.json` — `{query_uuid: {query, type, source}}`; `type` is abstractive/extractive,
    `source` is text / text-image / text-table / text-table-image.
  - `qrels.json` — `{query_uuid: {doc_id, section_id}}`, the relevance judgments. Section-level:
    this is why chunks carry `section_id`.
  - `answers.json` — `{query_uuid: reference answer}`.
  - `pdf_urls.json` — `{doc_id: arXiv PDF url}` for all 1,000 papers.
  - `corpus/{doc_id}.json` — the parsed paper: title, authors, categories, abstract, and
    `sections[]` with text, markdown tables and base64 images.
- **`arxiv_manifest.json`** — written by `scripts/run_ingestion.py` after each paper:
  `{paper_id: {title, authors, categories, published, updated, n_chunks}}`. Doubles as the
  resume log — a paper listed here with `n_chunks` is skipped on re-run unless `--force`.
- **`eval/results_<timestamp>.json`** — evaluation output, with the full pipeline config recorded
  alongside the scores so runs can be compared.
- **`state/`** — LangGraph SQLite checkpoints when `CHECKPOINT_BACKEND=sqlite`.

There is no hand-written `golden_qa.json` any more, and that is the point: the evaluation set is
built at run time from the benchmark's own queries and qrels (`evaluation/benchmark.py`), filtered
to the papers you actually indexed.
