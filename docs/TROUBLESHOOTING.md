# Troubleshooting

Start with **http://127.0.0.1:8000/health**. It reports each component separately, so it usually
points straight at the cause.

## Backend

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on start | Running the system Python instead of the virtualenv | `.venv\Scripts\python.exe scripts/run_api.py` (Windows) or activate `.venv` first |
| `ValidationError … azure_openai_… Field required` | `backend/.env` missing, or the command was run from the wrong folder | `cd backend`, then `copy .env.example .env` and fill it in |
| `Port 8000 is already in use` | An old API process is still running | See [PORTS_AND_SERVICES.md](PORTS_AND_SERVICES.md#port-already-in-use), or use `--port 8001` |
| First start hangs for ~30 s | The cross-encoder is loading. On the very first run it is also downloaded from Hugging Face. | Wait. Later starts are faster. |
| `/health` → `vector_store: error` | Qdrant is not running, or `QDRANT_URL` is wrong | `docker compose up -d`, then check http://localhost:6333/dashboard |
| `/health` → `llm: error` | Wrong Azure endpoint, key or deployment name | Use the **base** URL only. Check the deployment name in Azure AI Foundry. |
| `/health` → `cache: degraded` | Redis is not running | `docker compose up -d`. Without it everything still works, just slower. |
| `/health` → `auth: degraded` | `API_KEYS` is empty | Expected locally. Set it before exposing the API. |
| 429 responses | Per-client or global rate limit reached | Wait for `Retry-After`, or raise the `RATE_LIMIT_*` values |
| Answer says the paper "is not reported in the retrieved context" | The paper is not indexed | Check the Corpus tab or `GET /v1/corpus`. Ingest more papers. |
| Follow-up questions lose context | Several workers with in-memory checkpoints | `CHECKPOINT_BACKEND=sqlite`, or run one worker |
| `UnicodeEncodeError` in a Windows console | cp1252 console encoding | The scripts already force UTF-8. If you call modules directly, set `PYTHONIOENCODING=utf-8`. |
| Evaluation falls back to the built-in judge | RAGAS is not installed | `pip install -r requirements-eval.txt` |

## Frontend

| Symptom | Cause | Fix |
|---|---|---|
| "API unreachable" badge | The backend is not running, or `VITE_API_BASE_URL` is wrong | Start the API. Check `.env.local`, then restart `npm run dev`. |
| CORS error in the browser console | Backend `CORS_ORIGINS` does not include the UI origin | Add `http://localhost:5173` (or your origin) to `CORS_ORIGINS` and restart the API |
| 401 on every request | Backend has `API_KEYS` set | Put a matching key in `VITE_API_KEY` and restart `npm run dev` |
| Change to `.env.local` has no effect | Vite reads env files only at startup | Restart `npm run dev` |
| Corpus tab empty | Nothing indexed yet | Run `download_corpus.py`, then `run_ingestion.py` |
| `npm install` fails or Vite will not start | Node is too old | Node 20.19+ or 22.12+ |

## Infrastructure

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose up` says port 6333 or 6379 is allocated | An older container started with `docker run` | `docker ps` to find it, then `docker stop <name>`, or keep using it and skip compose |
| Index disappeared after a restart | Container was started without a volume, or removed with `down -v` | Use `docker compose`, which keeps named volumes, and re-run ingestion |
| Ingestion stopped halfway | Network error or Azure rate limit | Re-run `run_ingestion.py`. It resumes from where it stopped. |
