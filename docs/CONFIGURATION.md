# Configuration

Both apps are configured only through environment files, which are **never committed**.

| App | Template (committed) | Your copy (git-ignored) | Read by |
|---|---|---|---|
| Backend | `backend/.env.example` | `backend/.env` | `backend/config/settings.py`. Nothing else reads `os.environ`. |
| Frontend | `frontend/.env.example` | `frontend/.env.local` | Vite, at **build time** (`import.meta.env`) |

A real environment variable always overrides the value in the file.

## Backend

**Required.** The API will not start without these:

| Variable | Example | Notes |
|---|---|---|
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | `https://<resource>.cognitiveservices.azure.com` | Base URL only, with no path or query string |
| `AZURE_OPENAI_EMBEDDING_API_KEY` | — | Secret |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | Your deployment name |
| `AZURE_OPENAI_CHAT_ENDPOINT` | `https://<resource>.openai.azure.com` | Can be a different resource from embeddings |
| `AZURE_OPENAI_CHAT_API_KEY` | — | Secret |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o-mini` | Your deployment name |

**Everything else has a working default.** Grouped by purpose:

| Group | Variables | Defaults |
|---|---|---|
| API versions | `AZURE_OPENAI_EMBEDDING_API_VERSION`, `AZURE_OPENAI_CHAT_API_VERSION` | `2023-05-15`, `2025-01-01-preview` |
| Qdrant | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME` | `http://localhost:6333`, empty, `arxiv_papers` |
| Corpus | `HF_DATASET_REPO`, `HF_DATASET_SUBDIR`, `OPEN_RAGBENCH_DIR`, `CORPUS_MANIFEST_PATH` | `vectara/open_ragbench`, `pdf/arxiv`, `data/open_ragbench`, `data/arxiv_manifest.json` |
| Ingestion | `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_PARAGRAPHS`, `SKIP_REFERENCE_SECTIONS`, `CAPTION_FIGURES` | `400`, `1`, `true`, `false` |
| Vision model | `VLM_API_BASE`, `VLM_API_KEY`, `VLM_MODEL_NAME` | Ollama at `http://localhost:11434/v1`, used only with `CAPTION_FIGURES=true` |
| Retrieval | `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `RERANKER_MODEL` | `20`, `5`, `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Techniques | `ENABLE_QUERY_REWRITE`, `ENABLE_HYDE`, `ENABLE_RERANKER`, `ENABLE_PLANNER`, `MAX_SUB_QUESTIONS`, `ENABLE_CONTEXT_GRADING`, `ENABLE_GROUNDEDNESS_CHECK`, `MAX_RETRIEVAL_RETRIES` | all `true`; `3`; `1` |
| Memory | `MEMORY_MAX_TURNS`, `CHECKPOINT_BACKEND`, `CHECKPOINT_PATH`, `CLASSIFIER_MIN_CONFIDENCE` | `6`, `memory`, `data/state/checkpoints.sqlite`, `0.5` |
| Guardrails | `PII_REDACTION_ENABLED`, `PII_BACKEND`, `INJECTION_GUARD_ENABLED`, `SELF_HARM_GUARD_ENABLED`, `HAZARD_GUARD_ENABLED`, `REFERENCE_CHECK_ENABLED`, `MAX_INPUT_CHARS` | all `true`; `regex`; `1500` |
| Rate limits & resilience | `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_HOUR`, `GLOBAL_RATE_LIMIT_PER_MINUTE`, `LLM_MAX_RETRIES`, `LLM_MAX_CONCURRENCY`, `LLM_TIMEOUT_SECONDS` | `10`, `100`, `60`, `4`, `4`, `60` |
| Live scoring | `ENABLE_INFERENCE_SCORING` | `true`. This only allows a request to opt in with `"score": true`. It never scores every request. |
| HTTP | `API_KEYS`, `CORS_ORIGINS`, `TRUST_FORWARDED_FOR` | empty (**no auth**), `*`, `false` |
| Cache | `REDIS_URL`, `REDIS_CACHE_ENABLED` | `redis://localhost:6379/0`, `true` |
| Logging | `LOG_LEVEL` | `INFO` |

`backend/.env.example` documents each variable inline. `backend/config/settings.py` is the authoritative list.

Most technique toggles can also be overridden **per request** through `options` in `POST /v1/chat`, and
from the frontend's **Pipeline** menu. That makes A/B comparisons possible without a restart.

## Frontend

| Variable | Default | Notes |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Where the backend listens |
| `VITE_API_KEY` | empty | Sent as `X-API-Key`. Needed only when the backend sets `API_KEYS`. |

> **`VITE_*` values are compiled into the JavaScript bundle.** Anyone who loads the built site can read
> `VITE_API_KEY`. Use it for local development keys only. A real deployment should put the API behind a
> server-side proxy that holds the key.

Restart `npm run dev` after changing `.env.local`.

## Production checklist

- [ ] `API_KEYS` set (comma-separated, long random values)
- [ ] `CORS_ORIGINS` set to the exact frontend origin, not `*`
- [ ] `TRUST_FORWARDED_FOR=true` **only** if a trusted reverse proxy sets `X-Forwarded-For`
- [ ] `CHECKPOINT_BACKEND=sqlite` (or a shared store) if you run more than one worker
- [ ] `QDRANT_API_KEY` set if Qdrant is reachable from outside the host
- [ ] Rate limits sized to your Azure quota
- [ ] `/health` shows `auth: ok`
