# Getting started

From a fresh clone to a working answer in the browser. Each step ends with a **check** so you know
it worked before moving on.

## 0 · Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | **3.11** (pinned in `backend/.python-version`) | Backend |
| Node.js | **20.19+ or 22.12+** (22 LTS recommended) | Frontend (Vite 8 requirement) |
| Docker Desktop / Docker Engine | any recent | Qdrant and Redis |
| Git | any recent | — |
| Azure OpenAI | one chat deployment + one embedding deployment | LLM calls and embeddings |

Disk: about 2–3 GB for the Python environment (PyTorch is pulled in by the cross-encoder reranker),
plus about 2 GB if you download the full judged corpus.

## 1 · Clone

```bash
git clone https://github.com/Bilal-73/ResearchRAG.git
```

```bash
cd ResearchRAG
```

## 2 · Start the infrastructure

```bash
docker compose up -d
```

This starts **Qdrant** (vector store, ports 6333/6334) and **Redis** (cache, port 6379), with named
volumes so the index and cache survive restarts.

**Check:** open http://localhost:6333/dashboard. The Qdrant dashboard should load.
`docker compose ps` should show both containers as `running`.

## 3 · Set up the backend

All backend commands run **from inside `backend/`**.

```bash
cd backend
```

```bash
python -m venv .venv
```

Activate it. On Windows: `.venv\Scripts\activate`. On macOS/Linux: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

Create your environment file. On Windows: `copy .env.example .env`. On macOS/Linux:
`cp .env.example .env`. Then open `.env` and fill in at least:

```
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://<resource>.cognitiveservices.azure.com
AZURE_OPENAI_EMBEDDING_API_KEY=...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_CHAT_API_KEY=...
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

Use the **base** resource URL only, with no `/openai/deployments/...` path. Every other variable has a
working default. See [CONFIGURATION.md](CONFIGURATION.md).

**Check:** `python -m pytest tests -q` (after `pip install -r requirements-dev.txt`). It is fully
offline and should report 250+ passed.

> **Windows tip:** if you get `ModuleNotFoundError`, you are running the system Python, not the
> virtualenv. Call it explicitly: `.venv\Scripts\python.exe scripts/run_api.py`.

## 4 · Download and index papers

```bash
python scripts/download_corpus.py --limit 50
```

```bash
python scripts/run_ingestion.py
```

`--limit 50` takes a few minutes and is enough to try everything. Drop the flag to get the ~396 judged
papers (benchmark-complete), or pass `--all` for all 1,000 including hard negatives. Ingestion is
resumable: re-running it skips papers that are already indexed.

**Check:** the Qdrant dashboard now shows an `arxiv_papers` collection with points in it, and
`data/arxiv_manifest.json` lists the indexed papers.

## 5 · Run the API

```bash
python scripts/run_api.py
```

The first start takes ~30 seconds while the cross-encoder loads.

**Check:** http://127.0.0.1:8000/health. `agent`, `vector_store` and `llm` should be `ok`. `cache` is
`ok` with Redis running. `auth: degraded` is expected locally, because no API keys are set.

Try one question from the terminal:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"What problem does the VDF-RLS algorithm solve?\"}"
```

## 6 · Run the frontend

In a **second terminal**, from the repository root:

```bash
cd frontend
```

```bash
npm install
```

```bash
npm run dev
```

Optional: copy `.env.example` to `.env.local` if the API is not on `http://127.0.0.1:8000` or needs a
key.

**Check:** http://localhost:5173. The health strip at the top should be green. Ask a question from
[backend/docs/TEST_QUESTIONS.md](../backend/docs/TEST_QUESTIONS.md) and click an `[S1]` citation.

## 7 · (Optional) Evaluate

```bash
pip install -r requirements-eval.txt
```

```bash
python scripts/run_evaluation.py -n 25
```

Results are written to `backend/data/eval/results_<timestamp>.json`. A full 100-query run costs
roughly 700–1,000 LLM calls. See the Evaluation section of [backend/README.md](../backend/README.md).

## Stopping everything

`Ctrl+C` in the API and frontend terminals, then from the repository root:

```bash
docker compose down
```

Your index and cache are kept in Docker volumes. `docker compose down -v` deletes them.
