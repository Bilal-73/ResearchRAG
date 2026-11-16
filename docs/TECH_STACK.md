# Tech stack

## Backend (`backend/`)

| Concern | Technology | Version | Why it is here |
|---|---|---|---|
| Language | Python | 3.11 | Pinned in `.python-version`. Best-supported version across LangChain, PyTorch and RAGAS. |
| HTTP API | **FastAPI** + **Uvicorn** | 0.115+ / 0.30+ | Typed request/response models, automatic OpenAPI and Swagger, async. |
| Validation & config | **Pydantic** v2, **pydantic-settings** | 2.7+ / 2.2+ | Every setting is typed and validated once, at startup, from `.env`. |
| Agent orchestration | **LangGraph** | 1.x | Explicit state machine with conditional edges, bounded retries and checkpointed memory. |
| LLM primitives | **LangChain Core**, **langchain-openai** | 1.x | `ChatPromptTemplate`, LCEL chains, `Embeddings` / `VectorStore` / `BaseRetriever` interfaces. |
| Chat model | Azure OpenAI **gpt-4o-mini** | API 2025-01-01-preview | Classify, rewrite, HyDE, plan, grade, generate and verify. Cheap and fast enough for 4–8 calls per turn. |
| Embeddings | Azure OpenAI **text-embedding-3-small** | API 2023-05-15 | Dense vectors for chunks and queries. |
| Sparse vectors | Hashing encoder (in-repo) | — | Lexical signal for hybrid search without a fitted vocabulary. |
| Vector database | **Qdrant** (`qdrant-client`) | 1.10+ | Native hybrid dense + sparse search and payload filters. Runs in Docker. Has an in-memory mode used by tests. |
| Reranker | **sentence-transformers** cross-encoder `ms-marco-MiniLM-L-6-v2` | 2.7+ | Local CPU reranking of the fused candidates. Pulls in PyTorch. |
| Cache | **Redis** (`redis-py`) | 5+ | Embedding and caption cache with hit/miss instrumentation. |
| Conversation memory | LangGraph checkpointer (memory or **SQLite**) | `langgraph-checkpoint-sqlite` 3.x | Per-conversation state. SQLite survives restarts and multiple workers. |
| Tokenisation | **tiktoken** (`cl100k_base`) | 0.7+ | Exact token counts when packing chunks. |
| HTTP client | **httpx** | 0.27+ | Corpus download and FastAPI's `TestClient`. |
| Evaluation | **RAGAS**, **datasets** | 0.3.x | Faithfulness, answer relevancy, context precision/recall. Optional install. |
| PII (optional) | **Presidio** + spaCy `en_core_web_sm` | 2.2+ | Detects names, not just regex patterns. `PII_BACKEND=presidio`. |
| Figure captions (optional) | Any OpenAI-compatible VLM, e.g. **Qwen2.5-VL-7B** via **Ollama** | — | Reads axes and trends off figures. `CAPTION_FIGURES=true`. |
| Tests | **pytest** | 8+ | 250+ offline tests. LangChain fakes stand in for Azure, and in-memory Qdrant for the vector store. |

Requirement files are layered, so you install only what you need:

| File | Adds | Install when |
|---|---|---|
| `requirements.txt` | the runtime | always |
| `requirements-dev.txt` | pytest | running tests |
| `requirements-ingest.txt` | tqdm progress bars | ingesting large corpora |
| `requirements-eval.txt` | RAGAS, datasets | running `run_evaluation.py` with RAGAS |
| `requirements-extras.txt` | Presidio + spaCy model | `PII_BACKEND=presidio` |

## Frontend (`frontend/`)

| Concern | Technology | Version | Why it is here |
|---|---|---|---|
| UI library | **React** | 19 | Component model. Nothing else is needed for four tabs. |
| Language | **TypeScript** | 6 | `src/api/types.ts` mirrors the backend schemas, so contract drift becomes a compile error. |
| Build / dev server | **Vite** with `@vitejs/plugin-react` | 8 | Instant HMR in development, optimised static bundle for production. |
| Linting | **oxlint** | 1.x | Fast Rust-based linter, configured in `.oxlintrc.json`. |
| Styling | Plain CSS with design tokens (`src/index.css`) | — | Dark by default, light via `prefers-color-scheme`. No CSS framework. |
| State | `useState` + a typed `fetch` client | — | No Redux or React Query. One data source does not need them. |

## Infrastructure and tooling

| Concern | Technology |
|---|---|
| Local services | Docker Compose: `qdrant/qdrant` and `redis` (see `docker-compose.yml`) |
| CI | GitHub Actions: backend tests and frontend lint + build (see `.github/workflows/ci.yml`) |
| Dataset | [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench), arXiv split: 1,000 papers, 3,045 queries, section-level qrels |

## Techniques implemented

Query rewriting · HyDE (Hypothetical Document Embeddings) · multi-query retrieval · question
decomposition / planning · hybrid dense + sparse search · reciprocal-rank fusion · cross-encoder
reranking · corrective retrieval (context grading with retry) · groundedness verification ·
citation enforcement · fabricated-reference stripping · conversation memory with rolling summary ·
embedding caching · multimodal indexing (text, tables, figures) · inference-time RAGAS scoring ·
stratified benchmark evaluation · refusal testing on deliberately unindexed evidence.
