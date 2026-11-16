# Research RAG

**An agentic, guardrailed, benchmark-evaluated Retrieval-Augmented Generation system over arXiv research papers — with a FastAPI backend and a React inspection UI.**

Ask a question about the indexed papers and get an answer where every claim is cited down to the
**paper section** it came from. Anything the corpus does not support is refused instead of guessed.
Retrieval quality is *measured* against the [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench),
not asserted.

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-6-3178C6?logo=typescript&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## What is in this repository

This is a **monorepo** with two applications that run as separate processes and talk over HTTP.

| Folder | What it is | Language / stack | Runs on |
|---|---|---|---|
| [`backend/`](backend/) | The RAG system: ingestion, retrieval, LangGraph agent, guardrails, evaluation, and the **HTTP API** | Python 3.11 · FastAPI · LangChain · LangGraph · Qdrant · Redis | **http://127.0.0.1:8000** |
| [`frontend/`](frontend/) | The browser UI: chat with clickable citations, agent trace, cache and memory inspector, corpus browser, guardrail catalogue | React 19 · TypeScript · Vite | **http://localhost:5173** |
| [`docs/`](docs/) | Repository-wide documentation: architecture, ports, tech stack, setup, conventions | Markdown | — |
| [`docker-compose.yml`](docker-compose.yml) | The two infrastructure services the backend needs (Qdrant, Redis) | Docker | 6333 / 6334 / 6379 |

```
ResearchRAG/
├── backend/                 Python API — start here for the RAG pipeline
│   ├── config/              settings.py: every environment variable in one place
│   ├── src/research_rag/    the package (api, agents, retrieval, guardrails, evaluation, ...)
│   ├── scripts/             download_corpus, run_ingestion, run_api, run_evaluation, ...
│   ├── tests/               250+ offline tests (no Azure or Qdrant needed)
│   ├── docs/                system guide, responsible-AI notes, test questions, demo script
│   └── README.md            backend deep-dive: API, status codes, evaluation, costs
├── frontend/                React client — start here for the UI
│   ├── src/api/             typed API client + types mirroring the backend schemas
│   ├── src/components/      Chat, Context & Cache, Corpus, Guardrails views
│   └── README.md            frontend deep-dive: tabs, configuration, troubleshooting
├── docs/                    repository-wide documentation (index: docs/README.md)
├── docker-compose.yml       Qdrant + Redis for local development
├── .github/                 CI workflow and pull-request template
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE                  MIT
```

Every backend module folder also has its own `README.md` explaining that module in isolation.

---

## Ports at a glance

| Port | Service | Started by | Required? |
|---|---|---|---|
| **8000** | Backend API (FastAPI / Uvicorn) — Swagger at `/docs`, health at `/health` | `python scripts/run_api.py` in `backend/` | Yes |
| **5173** | Frontend dev server (Vite) | `npm run dev` in `frontend/` | For the UI |
| 4173 | Frontend production preview (`vite preview`) | `npm run preview` in `frontend/` | No |
| **6333** | Qdrant REST API + dashboard (`/dashboard`) | `docker compose up -d` | Yes |
| 6334 | Qdrant gRPC | `docker compose up -d` | No (exposed for tooling) |
| 6379 | Redis — embedding / caption cache | `docker compose up -d` | No — the API fails open without it |
| 11434 | Ollama — vision model for figure captions | `ollama serve` | Only if `CAPTION_FIGURES=true` |

Full details, how to change each one, and how to free a busy port: [docs/PORTS_AND_SERVICES.md](docs/PORTS_AND_SERVICES.md).

---

## Quick start

**Prerequisites:** Python 3.11, Node.js 20.19+ (22 LTS recommended), Docker, and an Azure OpenAI
resource with a chat deployment (e.g. `gpt-4o-mini`) and an embedding deployment
(`text-embedding-3-small`).

**1 · Infrastructure** (from the repository root):

```bash
docker compose up -d
```

**2 · Backend** (in `backend/`):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env — then fill in your Azure keys
python scripts/download_corpus.py --limit 50
python scripts/run_ingestion.py
python scripts/run_api.py
```

**3 · Frontend** (in a second terminal):

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** for the UI, or **http://127.0.0.1:8000/docs** for the interactive API.

> Run backend commands **from inside `backend/`**. The `.env` file and the `data/` paths are
> resolved relative to the working directory.

Step-by-step version with verification checks after each step: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

---

## How it works, in one picture

```
 Browser (React, :5173)
        │  fetch  POST /v1/chat, GET /v1/corpus, GET /v1/cache, ...   (X-API-Key optional)
        ▼
 FastAPI (:8000) ── rate limit · auth · request id
        │
        ▼
 LangGraph agent
   guard_input → classify → rewrite_query → plan → HyDE → retrieve → grade → generate → output_guard
        │                                   │                             │
        │                         Qdrant (:6333)                 Azure OpenAI (chat)
        │                   hybrid dense + sparse search          answers, judges, rewrites
        │                                   │
        └──── Redis (:6379) caches embeddings ◄── Azure OpenAI (embeddings)
```

The full architecture, the agent graph, and the request lifecycle are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). An interactive diagram is in
[`docs/architecture-diagram.html`](docs/architecture-diagram.html) (download and open in a browser).

---

## Highlights

- **Section-level citations.** Every answer carries `[S#]` markers that resolve to a paper, a section
  and a chunk type (text, table or figure).
- **Agentic retrieval.** Intent routing, follow-up resolution, query rewriting, HyDE, multi-part
  question planning, context grading with one corrective retry, and a groundedness check.
- **19 guardrails** across input, retrieval and output: PII redaction, prompt-injection defence,
  dual-use screening, crisis-language handling, citation enforcement, and stripping of invented
  arXiv ids and DOIs.
- **Measured, not asserted.** Recall, MRR and NDCG at document *and* section level, split by modality;
  RAGAS for answer quality; a refusal test built from real questions whose evidence was deliberately
  left out.
- **Observable.** The UI shows the agent's route and timings, the guardrails that fired, the Redis hit
  rate, and the conversation memory the next prompt will be built from.

---

## Documentation map

| If you want to… | Read |
|---|---|
| Run it for the first time | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Understand how the pieces fit | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Know which port does what | [docs/PORTS_AND_SERVICES.md](docs/PORTS_AND_SERVICES.md) |
| See every library and why it was picked | [docs/TECH_STACK.md](docs/TECH_STACK.md) |
| Configure it (every environment variable) | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| Follow the project's rules | [docs/DOS_AND_DONTS.md](docs/DOS_AND_DONTS.md) |
| Fix something that is broken | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| See what could be built next | [docs/ROADMAP.md](docs/ROADMAP.md) |
| Use the API in depth | [backend/README.md](backend/README.md) |
| Understand the UI in depth | [frontend/README.md](frontend/README.md) |
| Read the full design story | [backend/docs/SYSTEM_GUIDE.md](backend/docs/SYSTEM_GUIDE.md) |
| Review the safety posture | [backend/docs/RESPONSIBLE_AI.md](backend/docs/RESPONSIBLE_AI.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Report a vulnerability | [SECURITY.md](SECURITY.md) |

---

## Author

**Bilal Imran** — design, implementation and evaluation of both the backend and the frontend.
GitHub: [@Bilal-73](https://github.com/Bilal-73)

## License

[MIT](LICENSE) © 2026 Bilal Imran.

The indexed papers come from arXiv via the Open RAG Benchmark (`vectara/open_ragbench`) and stay under
their authors' own licenses. This repository contains no paper content. The corpus is downloaded
locally by `scripts/download_corpus.py` and is git-ignored.
