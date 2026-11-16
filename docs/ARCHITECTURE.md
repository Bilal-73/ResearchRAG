# Architecture

Research RAG is two applications and three external dependencies.

```
┌──────────────────────────┐        HTTP / JSON         ┌───────────────────────────────────────────┐
│  frontend/  (:5173)      │  ───────────────────────►  │  backend/  FastAPI (:8000)                │
│  React 19 + TypeScript   │  POST /v1/chat             │                                           │
│  Vite dev server         │  GET  /v1/corpus           │  api/        routers · schemas · auth     │
│                          │  GET  /v1/cache            │     │                                     │
│  Chat · Context & Cache  │  GET  /v1/guardrails       │     ▼                                     │
│  Corpus · Guardrails     │  GET  /health              │  agents/     LangGraph state machine      │
└──────────────────────────┘  ◄─────────────────────── │     │                                     │
                               answer, sources,         │     ├─► retrieval/   hybrid · HyDE · RRF  │
                               guardrail events,        │     │                cross-encoder rerank │
                               diagnostics trace        │     ├─► generation/  citation-only prompt │
                                                        │     └─► guardrails/  input · output       │
                                                        └──────┬───────────┬───────────┬────────────┘
                                                               │           │           │
                                                       Qdrant :6333   Redis :6379   Azure OpenAI
                                                       vectors        embedding     chat + embeddings
                                                       (required)     cache (opt.)  (required)
```

An interactive version is in [architecture-diagram.html](architecture-diagram.html).

## Backend layers

All code is in `backend/src/research_rag/`.

| Layer | Package | Responsibility |
|---|---|---|
| HTTP | `api/` | FastAPI app, routers (`chat`, `context`, `corpus`, `meta`), Pydantic schemas, API-key auth, CORS, request ids, uniform error bodies |
| Agent | `agents/` | LangGraph `StateGraph`: intent classification, routing, planning, memory, the node implementations |
| Retrieval | `retrieval/` | LangChain `BaseRetriever`s: multi-query hybrid search, HyDE, reciprocal-rank fusion, cross-encoder reranker (`BaseDocumentCompressor`) |
| Generation | `generation/` | Citation-enforcing `ChatPromptTemplate` and the answer chain |
| LLM | `llm/` | Model factories (`AzureChatOpenAI`, `AzureOpenAIEmbeddings`) and a resilient client with retries, back-off and a concurrency cap |
| Embeddings | `embeddings/` | Redis-cached `Embeddings` wrapper and a hashing sparse encoder |
| Vector store | `vectorstore/` | Qdrant as a LangChain `VectorStore` with hybrid dense + sparse search |
| Guardrails | `guardrails/` | PII, injection, safety, rate limiting, output checks, and the **single registry** the UI reads |
| Ingestion | `ingestion/` | Benchmark JSON reader, section-aware chunker, optional VLM captioner, resumable pipeline |
| Evaluation | `evaluation/` | Stratified eval sets, retrieval metrics, RAGAS, built-in LLM judge, refusal test |
| Corpus | `corpus/` | Live view of what is indexed |
| Caching | `caching/` | Instrumented Redis cache (hit/miss counters, latency) |
| Composition | `bootstrap.py` | **The one place concrete implementations are chosen and wired together** |
| Config | `backend/config/settings.py` | **The one place environment variables are read** |

Swapping a component (a different vector store, reranker or embedding model) is a change in
`bootstrap.py` and nowhere else.

## The agent graph

```
guard_input ─┬─(blocked)──────────────────────────────────► finalize
             └► classify ─┬─ chit_chat ──► respond_chitchat ─► finalize
                          ├─ out_of_scope ► respond_scope ───► finalize
                          └─ follow_up | new_question
                                ▼
                     rewrite_query ► plan ─┬─(compound)► retrieve_plan ─┐
                           ▲               └─(single)──► hyde ► retrieve ┤
                           │                                             ▼
                           │                            ┌──(no hits)► respond_not_found
                           │                            └► grade ─┬─(weak, retries left)─┐
                           └──────────────────────────────────────┘                      │
                                                              (ok / out of retries)► generate
                                                                                      ▼
                                                                          output_guard ► finalize
```

- **Memory** is per `conversation_id` through a LangGraph checkpointer. The last `MEMORY_MAX_TURNS` turns are kept
  verbatim and older turns are folded into a rolling summary. Only PII-redacted text is ever stored.
  `CHECKPOINT_BACKEND=sqlite` keeps conversations across restarts and workers.
- **Retries are bounded** (`MAX_RETRIEVAL_RETRIES`), so the graph cannot loop forever.
- Every node records its name, role and duration into `diagnostics.trace`. That is what the UI's
  inspector shows.

## Lifecycle of one chat request

1. The browser `POST`s `/v1/chat` with `message`, an optional `conversation_id`, per-request `options` and `score`.
2. Middleware assigns or echoes `X-Request-ID` and starts the response timer. The auth dependency checks
   `X-API-Key` when `API_KEYS` is set. The rate limiter checks per-client and global budgets.
3. `guard_input`: length cap, PII redaction, crisis-language check, dual-use screening, injection scan.
4. `classify` picks an intent. Chit-chat and out-of-scope messages are answered without retrieval.
5. `rewrite_query` turns a follow-up into a standalone query. `plan` splits compound questions.
   `hyde` drafts a hypothetical passage to embed.
6. `retrieve`: dense + sparse search in Qdrant, fused with RRF, reranked by a local cross-encoder.
   Retrieved chunks are scanned for injected instructions.
7. `grade` decides whether the evidence is enough. If it is not, the graph rewrites the query and retrieves once more.
8. `generate` answers **only** from the evidence, citing `[S#]` for every claim.
9. `output_guard`: citation enforcement, fabricated arXiv id / DOI stripping, groundedness check, output PII scan.
10. The API returns **200** with `answer`, `answer_kind`, `sources`, `guardrail_events` and `diagnostics`. If
    `score: true`, RAGAS faithfulness and answer relevancy are also computed and returned.

## Ingestion path (offline)

```
download_corpus.py ─► data/open_ragbench/corpus/<arxiv-id>.json   (pre-parsed sections, tables, base64 figures)
run_ingestion.py   ─► section-aware chunker ─► text / table / figure chunks (each keeps section_id)
                   ─► embeddings (dense via Azure, sparse via hashing encoder, cached in Redis)
                   ─► Qdrant collection `arxiv_papers`  +  data/arxiv_manifest.json
```

The pipeline never downloads or parses PDFs. The benchmark ships each paper already parsed.

## Design decisions worth knowing

| Decision | Reason |
|---|---|
| **Refusals and safety blocks are HTTP 200** | The request was valid and the system behaved correctly. `answer_kind` carries the nuance, so a safety redirect never looks like a crash to a client. |
| **Section-level chunks and metrics** | The benchmark judges relevance at `(doc_id, section_id)`. Without section granularity you cannot tell "found the right paper" from "found the right paragraph". |
| **Evaluation runs the same agent the API serves** | `RAGSystem` wraps the agent, so you cannot accidentally score a pipeline no user ever hits. |
| **Caching fails open** | Redis is an optimisation, never a dependency. Without it, embeddings are recomputed and `/health` says `cache: degraded`. |
| **One guardrail registry** | The UI's Guardrails tab reads the same registry the pipeline uses, with flags taken from live settings. It cannot claim a guardrail is on while it is off. |
| **Frontend renders answers as text nodes** | Answers contain retrieved paper content, which is untrusted. `dangerouslySetInnerHTML` is never used. |
| **No UI framework or state library** | Four tabs and one data source: `useState` plus a typed fetch client is the whole architecture. |

## The API contract

`backend/src/research_rag/api/schemas.py` is the source of truth. FastAPI publishes it at
`/openapi.json`, and `frontend/src/api/types.ts` mirrors it by hand. **If you change a schema, change both
in the same commit.** The full endpoint and status-code tables are in
[backend/README.md](../backend/README.md#the-api).
