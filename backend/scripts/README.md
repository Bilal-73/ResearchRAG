# scripts/

Thin CLI entry points. All wiring lives in `research_rag/bootstrap.py` (inside the package, so the
API server shares it) — `_bootstrap.py` here only puts `src/` on the path and starts logging.

Run them in this order the first time:

1. **`download_corpus.py`** — mirrors the Open RAG Benchmark arXiv split into
   `data/open_ragbench/`. Metadata is ~1.6 MB; corpus files embed their figures as base64, so
   start with `--limit 50`. No flags fetches the ~396 *judged* papers (a benchmark-complete
   corpus); `--all` adds the 600 hard negatives.
2. **`run_ingestion.py`** — chunks and indexes into Qdrant. Resumable: re-running skips papers
   already in the manifest. `--dry-run` reports chunk counts without spending a single embedding
   call, which is the cheap way to tune `CHUNK_MAX_TOKENS`.
3. **`run_api.py`** — serves the FastAPI app; docs at `/docs`. Equivalent to
   `uvicorn research_rag.api.app:app --app-dir src`.
4. **`run_evaluation.py`** — the three-stage benchmark run (retrieval → answer quality → refusal).
   `--retrieval-only` costs nothing in LLM calls; results are written to `data/eval/`.

Also useful while developing:

- **`run_query.py`** — one question, no server: `python scripts/run_query.py "your question"`
  (add `--json` for the full result).
- **`run_chat.py`** — multi-turn terminal chat against the full agent (`/new`, `/memory`, `/quit`).
