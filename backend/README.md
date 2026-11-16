# Research RAG — Agentic, Guardrailed, Benchmark-Evaluated RAG over arXiv Papers

> **Backend** of the Research RAG monorepo. Repository overview, ports and setup: [../README.md](../README.md) ·
> Docs index: [../docs/](../docs/README.md) · Frontend: [../frontend/](../frontend/README.md)
> Run every command below **from inside this `backend/` folder**.

A multimodal Retrieval-Augmented Generation system over **arXiv research papers**, served as a
**FastAPI** HTTP API and evaluated against the
[Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) (`vectara/open_ragbench`,
arXiv split). Text, tables and figures are all retrievable, and every answer carries
system-generated citations down to the **paper section** — the same granularity the benchmark
judges, so retrieval quality is *measured*, not asserted.

On top of the retrieval core sits a **LangGraph agent** (intent routing, follow-up handling, query
rewriting, HyDE, question planning, corrective retrieval, memory) wrapped in **guardrails** (PII
redaction, prompt-injection defence, dual-use screening, crisis handling, citation enforcement,
fabricated-reference stripping, groundedness checking).

The whole data path is **one LangChain stack**: LangGraph orchestrates the agent, and every step inside
it is a LangChain primitive — `AzureChatOpenAI` models, `ChatPromptTemplate` prompts run as LCEL
chains, cached `Embeddings`, a Qdrant `VectorStore`, `BaseRetriever` retrievers and a
`BaseDocumentCompressor` reranker exchanging `Document`s. See `docs/SYSTEM_GUIDE.md` §5 for the map.

---

## Why this corpus

The previous version of this project answered questions about ten medical papers. That made for a
tidy demo and a useless evaluation: the "ground truth" was LLM-generated from the same chunks the
system retrieved, so the benchmark could only confirm the pipeline agreed with itself.

Open RAG Benchmark fixes exactly that, and brings the use case with it:

| | What it gives | Why it changes the system |
|---|---|---|
| **3,045 queries** with human-usable reference answers | ground truth nobody in this repo wrote | end-to-end scores mean something |
| **qrels at `(doc_id, section_id)`** | section-level relevance judgments | retrieval is scored on finding the right *section*, not just the right paper |
| **1,000 papers across every arXiv category** | a corpus too big and too broad to curate by hand | prompts and scope checks must summarise the corpus, not enumerate it |
| **32% of queries need a table or figure** | 763 text-image, 148 text-table, 220 text-table-image | multimodal indexing is load-bearing, and its contribution is separately measurable |
| **600 unjudged "hard negative" papers** | documents no query is about | realistic distractors, plus a principled unanswerable set for refusal testing |

**The use case:** a literature assistant for researchers working across unfamiliar fields — "what
does this paper actually claim, on what evidence, and where exactly does it say so?" Every claim
is traceable to a section; anything the corpus does not support is refused rather than guessed.

---

## What changed from the medical version

| Area | Before | Now |
|---|---|---|
| Interface | Streamlit app | **FastAPI** with OpenAPI docs, typed schemas, explicit status codes |
| Orchestration | LangGraph over raw `openai` / `qdrant-client` calls | **LangGraph + LangChain**: LCEL chains, `VectorStore`, `BaseRetriever`, `Document`s throughout |
| Corpus | 10 curated medical PDFs (Docling + OCR) | up to 1,000 arXiv papers from pre-parsed benchmark JSON (no PDF parsing) |
| Chunking | Docling `HybridChunker` | section-aware chunker that preserves `section_id` and splits out tables/figures |
| Ground truth | LLM-generated from the same chunks | the benchmark's own queries, qrels and reference answers |
| Retrieval metrics | Recall/MRR/NDCG at document level | **document *and* section level**, broken out by query type and modality |
| Refusal testing | hand-written "unanswerable" questions | real benchmark queries whose gold paper is deliberately unindexed |
| Safety guardrails | medical-advice redirect, emergency detection | **dual-use screening**, crisis handling, **fabricated-reference stripping** |
| Hallucination guard | medical disclaimer on dosing answers | arXiv ids and DOIs absent from the evidence are stripped and reported |

### Why the guardrails changed

The old safety layer was built for people asking about their own health. On an arXiv corpus the
risks are different, so three were replaced and one was added:

- **Removed — personal medical-advice redirect.** "Should I take this?" is now simply out of scope;
  there is no medical corpus to redirect it against.
- **Removed — emergency/self-harm medical detection**, narrowed to **crisis-language detection**.
  Any public chatbot meets people in distress, so the self-harm path stays; the chest-pain/ketones
  patterns went with the medical corpus.
- **Added — dual-use screening.** The corpus spans q-bio, chemistry and physics. *Discussing*
  hazardous research is the whole point of a literature assistant, so topic words never trigger the
  guard; it fires only on **procedural** requests ("how do I make/build …") aimed at a weapons-class
  category. "What detection method does this paper propose?" passes; "how do I synthesise …" does not.
- **Added — fabricated-reference check.** The signature hallucination on a scientific corpus is a
  confident, well-formatted, entirely invented arXiv id or DOI. Any identifier in an answer that
  appears nowhere in the retrieved evidence is stripped and reported as a guardrail event.

One existing rule also had to be relaxed: the injection scanner treated "ignore … previous …
context" as an attack, which is ordinary ML-paper prose, and it ran against retrieved chunks —
so it was silently dropping legitimate evidence. `context` is no longer a trigger word
(`guardrails/injection.py`).

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
copy .env.example .env                              # then fill in your Azure keys
docker compose up -d                                # from the repo root: Qdrant (6333) + Redis (6379)
```

```bash
python scripts/download_corpus.py --limit 50
```

```bash
python scripts/run_ingestion.py
```

```bash
python scripts/run_api.py
```

Then open **http://127.0.0.1:8000/docs**.

> **Windows:** use the virtualenv's interpreter explicitly —
> `.venv\Scripts\python.exe scripts/run_api.py`. A bare `python` usually resolves to a system
> install with none of these dependencies; the scripts detect that and tell you, but it is the
> single most common way to get a `ModuleNotFoundError` here.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"What problem does the VDF-RLS algorithm solve?\"}"
```

Start small: `--limit 50` downloads and indexes ~50 papers in a few minutes. Drop the flag for the
~396 judged papers (a benchmark-complete corpus), or pass `--all` for all 1,000 including the hard
negatives.

---

## The API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat` | Ask a question. Returns the answer, citations, guardrail events and a full diagnostics trace. |
| `DELETE` | `/v1/chat/{conversation_id}` | Forget a conversation. |
| `GET` | `/v1/corpus` | What is indexed: counts, category breakdown, paginated paper list. |
| `GET` | `/v1/corpus/{paper_id}` | One paper's metadata. |
| `GET` | `/v1/guardrails` | Every guardrail, its stage, and whether it is live. |
| `POST` | `/v1/chat` with `"score": true` | Same turn, plus live RAGAS scores for that answer. |
| `GET` | `/v1/cache` | Redis cache state: hit/miss counters, hit rate, keys per namespace, server info. |
| `POST` | `/v1/cache/reset-stats` | Zero the counters (keeps cached data) for a clean cold-vs-warm demo. |
| `GET` | `/v1/chat/{conversation_id}/memory` | What the agent remembers: verbatim turns, rolling summary, focus. |
| `GET` | `/health` | Per-component status (agent, vector store, LLM, cache, auth). |

### Status codes

Designed so a frontend can branch on the code alone:

| Code | Meaning |
|---|---|
| `200` | A turn was produced — **including refusals, redirects and safety responses**. The request was valid and the system behaved correctly. |
| `400` | Empty message. |
| `401` | `API_KEYS` is set and the key is missing or wrong. |
| `404` | Unknown paper id or route. |
| `413` | Message longer than `MAX_INPUT_CHARS`. |
| `422` | Body didn't match the schema. |
| `429` | Rate limited — `Retry-After` in both header and body. |
| `503` | Agent unbuilt, or the vector store / LLM is unreachable. **Never a fabricated answer.** |

A blocked or refused turn is not an HTTP error. Branch on `answer_kind`
(`rag`, `chitchat`, `out_of_scope`, `not_found`, `blocked`, `safety`) and the `blocked` flag.
Every 4xx/5xx body has the same shape:

```json
{ "error": "rate_limited", "message": "...", "retry_after_seconds": 42.0 }
```

Every response carries `X-Request-ID` (echoed if you send one) and `X-Response-Time-Ms`.

### A chat response

```jsonc
{
  "answer": "The VDF-RLS method reached an RMSPE of 2.3426% under excitation [S1], ...",
  "answer_kind": "rag",
  "conversation_id": "thread-9f2c1a7b4e01",
  "sources": [{
    "label": "S1", "paper_id": "2410.14077v2", "section": "A. Performance comparison",
    "section_id": 8, "chunk_type": "table", "cited": true, "score": 7.9,
    "url": "https://arxiv.org/abs/2410.14077"
  }],
  "blocked": false,
  "guardrail_events": [{ "guardrail": "reference_check", "action": "stripped", "detail": "..." }],
  "diagnostics": {
    "intent": "new_question", "standalone_query": "...", "grounded": true,
    "route": ["guard_input", "classify", "rewrite_query", "plan", "hyde", "retrieve", "grade",
              "generate", "output_guard", "finalize"],
    "explanation": ["🧭 **Router** chose **new_question** ..."],
    "trace": [{ "node": "retrieve", "role": "Retriever + Reranker", "seconds": 0.41 }]
  }
}
```

Pass `?include_diagnostics=false` to drop the trace. Per-request pipeline toggles
(`enable_hyde`, `enable_reranker`, `enable_planner`, …) go in `options` — useful for A/B demos.

**Auth:** `API_KEYS` empty means no auth, which is fine locally and wrong anywhere else.
`/health` reports which mode is live, so an open deployment can't go unnoticed.

---

## Frontend

The React + TypeScript client lives next to this folder, in **`../frontend`** (same repository):

```bash
cd ../frontend && npm install && npm run dev
```

It is an inspection tool rather than just a chat window — clickable `[S#]` citations, the agent's
route and timing trace per turn, the guardrails that fired, a live Redis cache panel (hit rate,
keys, cold-vs-warm reset), the conversation memory the next prompt is built from, the indexed
corpus, and per-request pipeline toggles for A/B-ing retrieval techniques. See its README.

`CORS_ORIGINS=*` covers `http://localhost:5173` by default; set a real origin before deploying.

---

## Scoring a live answer

`POST /v1/chat` accepts `"score": true`, which runs RAGAS over the answer it just produced and
returns it in `scores`:

```jsonc
{ "answer": "...", "scores": { "faithfulness": 0.92, "answer_relevancy": 0.81, "seconds": 4.1 } }
```

Only the **reference-free** metrics apply here — `context_precision` and `context_recall` compare
against a gold answer, which a real user's question does not have. Non-grounded turns (chit-chat,
refusals, blocks) are not scored at all and say why, since there is no retrieved context to score
against. A `null` score means the metric could not be computed, never zero.

It is off by default — measured at **~15-20s of extra latency per turn** on gpt-4o-mini, which
roughly doubles a typical answer time. Toggleable in the frontend's chat composer, and can be
disabled server-side with `ENABLE_INFERENCE_SCORING=false`.

---

## Context engineering (Redis)

Embeddings are deterministic for a given text, so identical chunks and repeated queries are served
from Redis instead of the embedding API. Cache behaviour is invisible from the outside, so it is
instrumented: `GET /v1/cache` reports hits, misses, hit rate, read/write latency and keys per
namespace, and `/health` carries a `cache` component.

```bash
docker run -d -p 6379:6379 --name redis-server redis
```

Without it the system still works — caching fails open, every embedding is just recomputed, and
`/health` reports `cache: degraded`.

Repeating a question will not reach a 100% hit rate: query rewriting and HyDE generate fresh text
each turn, so those embeddings are new every time. Only the deterministic parts recur, which makes
the hit rate a decent readout of how much of the pipeline is actually deterministic.

---

## Evaluation

```bash
python scripts/run_evaluation.py                  # 100 stratified queries, all three stages
python scripts/run_evaluation.py -n 25            # quick pass
python scripts/run_evaluation.py --retrieval-only # skip generation + judging
python scripts/run_evaluation.py --sources text-table text-table-image
python scripts/run_evaluation.py --no-rerank      # measure retrieval before reranking
```

Results land in `data/eval/results_<timestamp>.json`, with the full config recorded alongside the
scores so two runs can be compared honestly.

**1 · Retrieval** — Recall@k, MRR, NDCG at **document** and **section** level, split by query type
(abstractive/extractive) and modality (text / text-image / text-table / text-table-image).
The section-level number is the strict one. The modality split is where a multimodal pipeline earns
its keep: a text-only indexer scores near zero on table and image queries, and the breakdown makes
that visible instead of burying it in an average.

`--retrieval-only` skips generation, grading and groundedness, but it is **not free**: with the
default settings it still pays query rewriting and HyDE, which are 2 LLM calls per query. For a
genuinely zero-LLM retrieval measurement, turn those off for the run:

```bash
ENABLE_QUERY_REWRITE=false ENABLE_HYDE=false python scripts/run_evaluation.py --retrieval-only
```

That also measures something different — raw embedding quality, rather than the query pipeline the
chatbot actually uses — so it is worth running both and comparing.

**2 · Answer quality** — **RAGAS** by default (`faithfulness`, `answer_relevancy`,
`context_precision`, `context_recall`), with its judge models wired to this project's Azure
deployments. `--judge builtin` swaps in the dependency-free judge, which instead scores
*correctness* against the benchmark's reference answer and *citation quality* — an answer can be
perfectly faithful to the wrong section, and only correctness catches that. The built-in judge also
reports `refusal_rate` separately, so a system that games faithfulness by refusing everything is
immediately visible.

RAGAS needs `pip install -r requirements-eval.txt`; without it the run falls back to the built-in
judge rather than failing.

**3 · Refusal** — real benchmark questions whose gold paper was deliberately left unindexed. The
question is well-formed and on-topic, the evidence genuinely isn't there, and the only correct
behaviour is to say so. Anything else is reported as a hallucination, in full.

Two things the harness does on purpose:

- **Only scores queries whose gold document is indexed.** Otherwise you are measuring your
  ingestion budget, not your retriever. The count of dropped queries is reported.
- **Stratified sampling with a fixed seed.** A random 100-query sample is ~63% text-only and can
  contain a handful of table queries, so multimodal regressions hide in the noise. Sampling
  proportionally across (type, source) keeps every slice represented and makes runs comparable.

---

## Project layout

```
backend/
├── config/            settings.py — all environment/config in one place
├── data/
│   ├── open_ragbench/   downloaded corpus: queries/qrels/answers + corpus/*.json
│   ├── arxiv_manifest.json  written by ingestion: paper_id -> title, categories, chunk count
│   └── eval/            evaluation results
├── src/research_rag/
│   ├── api/            FastAPI app, routers (chat, context, corpus, meta), schemas, dependencies
│   ├── models/         Chunk, RAGResponse, SourceCitation
│   ├── ingestion/      benchmark corpus reader, section-aware chunker, resumable pipeline
│   ├── embeddings/     cached LangChain Embeddings + hashing sparse encoder
│   ├── vectorstore/    Qdrant as a LangChain VectorStore, hybrid (dense + sparse) search
│   ├── retrieval/      LangChain retrievers: multi-query hybrid, HyDE, RRF fusion, cross-encoder reranker
│   ├── generation/     citation-enforcing ChatPromptTemplate + answer chain
│   ├── llm/            LangChain model factories + resilient client that builds LCEL chains
│   ├── agents/         LangGraph graph: state, intent, planner, nodes, memory, routing
│   ├── guardrails/     PII, injection, rate limit, safety, output checks, registry
│   ├── corpus/         live view of what is indexed
│   ├── evaluation/     benchmark eval sets + retrieval metrics + LLM judge + refusal test
│   └── bootstrap.py    composition root — the one place implementations are chosen
├── scripts/           download_corpus, run_ingestion, run_api, run_evaluation, run_query, run_chat
└── tests/             200+ offline tests (fakes for Azure/Qdrant; real agent, guardrails, API)

../frontend/   React + TypeScript client (same repository, its own README)
```

Every subfolder has its own `README.md` explaining that module in isolation.

---

## Agent architecture

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

Memory is per-`conversation_id` via a LangGraph checkpointer: the last N turns verbatim, older ones
folded into a rolling summary, and only PII-redacted text is ever stored. Set
`CHECKPOINT_BACKEND=sqlite` to keep conversations across restarts (and across API workers).

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Fully offline — Azure and Qdrant are faked (as LangChain `BaseChatModel` / `Embeddings` /
`VectorStore` fakes, plus the real Qdrant store against qdrant-client's in-memory mode), everything
else is real: the actual LCEL chains, the actual LangGraph graph, the actual guardrails, the actual
FastAPI app via `TestClient`.

---

## Costs and timings

| Operation | Cost |
|---|---|
| Download 396 judged papers | free, ~2 GB, a few minutes |
| Ingest 50 papers | ~2–4k embedding calls, a few minutes |
| `--retrieval-only` evaluation | 2 LLM calls per query (rewrite + HyDE); zero if both are disabled |
| Full evaluation, 100 queries | ~700–1,000 LLM calls (agent + judge) |
| One `/v1/chat` turn | 4–8 LLM calls with all techniques on; 2 with them off |

Redis caching (`REDIS_CACHE_ENABLED=true`) makes re-ingestion and repeat evaluation substantially
cheaper by reusing embeddings for unchanged text.

---

## Known limitations

- **Sparse vectors use a hashing encoder**, not BM25 over the corpus, so lexical matching is
  approximate. Swapping in SPLADE or a real BM25 index is a change to `HashingSparseEncoder.encode`
  and nothing else.
- **Figure captions default to the paper's own caption text.** `CAPTION_FIGURES=true` adds a VLM
  pass that reads axes and trends off the image, which should raise text-image recall — worth
  measuring on your own hardware before paying for it at scale.
- **Conversation memory is per-process** unless `CHECKPOINT_BACKEND=sqlite`; with multiple API
  workers and the in-memory backend, follow-ups can land on a worker that has no history.
- **The LLM judge shares a model family with the generator**, which is a known source of
  self-preference bias. The retrieval metrics are judge-free, and `--judge builtin` offers a second
  opinion (different prompts and metrics, though still the same model family).
