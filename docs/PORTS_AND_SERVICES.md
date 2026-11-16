# Ports and services

Five processes can be involved. Two are the applications in this repository, two are infrastructure
started by `docker compose`, and one is an optional local model server.

| Port | Process | Protocol | Bound to | Required | Configured by |
|---|---|---|---|---|---|
| **8000** | Backend API — FastAPI on Uvicorn | HTTP | `127.0.0.1` | **Yes** | `scripts/run_api.py --host --port` |
| **5173** | Frontend dev server — Vite | HTTP | `localhost` | For the UI | `npm run dev -- --port <n>` |
| 4173 | Frontend preview of the production build | HTTP | `localhost` | No | `npm run preview -- --port <n>` |
| **6333** | Qdrant REST API + web dashboard | HTTP | all interfaces (Docker) | **Yes** | `docker-compose.yml`, backend `QDRANT_URL` |
| 6334 | Qdrant gRPC | gRPC | all interfaces (Docker) | No | `docker-compose.yml` |
| 6379 | Redis | RESP | all interfaces (Docker) | No — caching fails open | `docker-compose.yml`, backend `REDIS_URL` |
| 11434 | Ollama (OpenAI-compatible vision model) | HTTP | `localhost` | Only with `CAPTION_FIGURES=true` | backend `VLM_API_BASE` |

Outbound only (no local port): **Azure OpenAI** for chat and embeddings, **Hugging Face Hub** for the
corpus download and the reranker model on first use, and **arxiv.org** for links shown in the UI.

## Who talks to whom

```
browser ──► :5173  Vite dev server  (serves the React bundle only; it does not proxy the API)
browser ──► :8000  FastAPI          (direct cross-origin fetch, allowed by CORS_ORIGINS)
:8000   ──► :6333  Qdrant           (QDRANT_URL)
:8000   ──► :6379  Redis            (REDIS_URL)
:8000   ──► Azure OpenAI            (AZURE_OPENAI_*_ENDPOINT)
ingestion ──► :11434 Ollama         (VLM_API_BASE, only when CAPTION_FIGURES=true)
```

Because the browser calls the API **directly** (no dev proxy), two settings must agree:

- frontend `VITE_API_BASE_URL` must point at the API (default `http://127.0.0.1:8000`);
- backend `CORS_ORIGINS` must allow the frontend's origin (default `*`, which allows `http://localhost:5173`).

## Useful URLs

| URL | What you get |
|---|---|
| http://127.0.0.1:8000/docs | Swagger UI: try every endpoint interactively |
| http://127.0.0.1:8000/redoc | ReDoc rendering of the same OpenAPI spec |
| http://127.0.0.1:8000/openapi.json | The API contract. `frontend/src/api/types.ts` mirrors it. |
| http://127.0.0.1:8000/health | Per-component status: agent, vector store, LLM, cache, auth |
| http://localhost:5173 | The UI |
| http://localhost:6333/dashboard | Qdrant dashboard: collections, points, search playground |

## Changing a port

| To move… | Change |
|---|---|
| the API to 8080 | `python scripts/run_api.py --port 8080` **and** frontend `VITE_API_BASE_URL=http://127.0.0.1:8080` (restart `npm run dev`) |
| the UI to 3000 | `npm run dev -- --port 3000`, and if `CORS_ORIGINS` is not `*`, add `http://localhost:3000` to it |
| Qdrant | the host side of `ports:` in `docker-compose.yml` **and** `QDRANT_URL` in `backend/.env` |
| Redis | the host side of `ports:` in `docker-compose.yml` **and** `REDIS_URL` in `backend/.env` |

To expose the API on your network, use `--host 0.0.0.0`. Before you do that, set `API_KEYS` and a
specific `CORS_ORIGINS`. See [DOS_AND_DONTS.md](DOS_AND_DONTS.md).

## "Port already in use"

`run_api.py` checks the port before starting and says so clearly. To find what is holding it:

| OS | Command |
|---|---|
| Windows | `netstat -ano \| findstr :8000`, then `taskkill /F /PID <pid>` |
| macOS / Linux | `lsof -i :8000`, then `kill <pid>` |

For 6333 or 6379 it is usually an older container started with `docker run`. Check with `docker ps` and stop it,
or skip `docker compose` and reuse that container.
