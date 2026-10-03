# Research RAG — Frontend

React + TypeScript + Vite client for the Research RAG API. It is an **inspection tool**, not just a
chat window: every answer comes with the route the agent took, the guardrails that fired, the
evidence it used, and what it cached and remembered along the way.

```bash
npm install
npm run dev
```

The backend must be running first:

```bash
cd ../DiabetesMedicalRAGv2 && .venv\Scripts\python.exe scripts/run_api.py
```

Then open http://localhost:5173.

---

## Configuration

Copy `.env.example` to `.env.local`:

| Variable | Default | Notes |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Where the FastAPI backend listens. |
| `VITE_API_KEY` | *(empty)* | Only needed when the backend has `API_KEYS` set. Sent as `X-API-Key`. |

Vite only exposes variables prefixed `VITE_`, and they are **baked into the bundle at build time** —
so `VITE_API_KEY` is visible to anyone who opens the built JS. Fine for a local development key,
not fine for a production secret; a real deployment should put the API behind a session-authenticated
proxy instead.

### CORS

The backend sets `CORS_ORIGINS=*` by default, which covers `http://localhost:5173` out of the box.
Before deploying, set it to the real origin on the **backend**:

```
CORS_ORIGINS=https://your-frontend.example.com
```

The API also exposes `X-Request-ID` and `Retry-After` to the browser, so a failed request can be
matched to a server log line.

---

## The four tabs

**Chat** — the conversation, with an inspector on the right for the selected turn. `[S1]` markers in
an answer are clickable and scroll to the matching source.

A key detail: **refusals, out-of-scope redirects and safety blocks are not errors.** The API returns
200 for them because the request was valid and the system behaved correctly, so they render as normal
messages with a badge (`out of scope`, `not in corpus`, `blocked`, `safety response`). Only genuine
4xx/5xx produce an error card — with the stable `error` code and a hint about the fix.

**Context & Cache** — the Redis layer and the agent's memory, side by side. See below.

**Corpus** — what is actually indexed: papers, titles, authors, arXiv categories, chunk counts, and
links to the abstract and PDF on arXiv. Check here first when an answer says "not reported in the
retrieved context" — usually the paper simply is not indexed, which is a different problem from a
retrieval failure.

**Guardrails** — all 19 guardrails grouped by pipeline stage with their live on/off state. Served
from the backend's single registry with flags read from live settings, so it cannot claim a guardrail
is on while its setting is off.

---

## Inspecting context engineering

The **Context & Cache** tab is the reason this app has the shape it does.

### Redis cache

Embeddings are deterministic for a given text, so identical chunks and repeated queries should be
served from Redis rather than recomputed through the embedding API. The panel shows hit rate, hits,
misses, writes, average read/write latency, keys per namespace (`embed`, `caption`), and Redis's own
memory and uptime.

**Cold vs warm, in four steps:**

1. Hit **Reset counters** (zeroes the counters, keeps the cached data).
2. Go to **Chat**, ask any question.
3. Return here — misses went up, keys were written.
4. Ask the *exact same* question again, refresh: hits climb.

Repeating a question will **not** reach 100%. Query rewriting and HyDE generate fresh text every
turn, so those embeddings are new each time; only the deterministic parts recur. That makes the hit
rate a useful readout of how much of your pipeline is actually deterministic — and an argument for
caching the rewrite itself if you want it higher.

### Conversation memory

The other half of context engineering: what goes *into* the next prompt. The panel shows turns kept
verbatim (against `MEMORY_MAX_TURNS`), the rolling LLM summary older turns are folded into, and the
current focus used to resolve follow-ups like "which one performed best?". Only PII-redacted text is
ever stored, and the summary is what keeps prompt size flat however long a conversation runs.

### Pipeline toggles

The **Pipeline** dropdown in the header sends per-request `options` — query rewrite, HyDE, reranker,
planner, context grading, groundedness check — each `default` / `on` / `off`. Turn HyDE off, ask the
same question, and compare the trace timings and retrieved sources to see what it actually buys.

---

## Project layout

```
src/
├── api/
│   ├── types.ts      mirrors the backend's Pydantic schemas (GET /openapi.json is the contract)
│   └── client.ts     the only place this app talks to the API; ApiError vs NetworkError
├── components/
│   ├── ChatView.tsx       conversation + per-turn inspector
│   ├── AnswerText.tsx     turns [S#] markers into clickable chips (text nodes only, never HTML)
│   ├── SourceList.tsx     cited evidence vs merely retrieved
│   ├── Diagnostics.tsx    route, query engineering, guardrail events, timing trace
│   ├── ContextView.tsx    Redis cache + conversation memory
│   ├── CorpusView.tsx     indexed papers
│   └── GuardrailsView.tsx guardrail catalogue
├── App.tsx           tabs, health strip, pipeline toggles
└── index.css         design tokens; dark-first, light via prefers-color-scheme
```

No UI framework and no state library: four tabs and one data source, so `useState` plus a typed
fetch client is the whole architecture. Answer text is rendered as text nodes, never
`dangerouslySetInnerHTML` — it contains retrieved paper content, which is untrusted.

---

## Where are the PDFs?

Nowhere on disk, by design. The Open RAG Benchmark ships each paper as pre-parsed JSON — sections,
markdown tables, base64 figures — at `data/open_ragbench/corpus/<arxiv-id>.json` in the backend repo.
That JSON is what gets chunked, embedded and indexed; no PDF is ever downloaded. Every "PDF" link in
this app therefore points at `arxiv.org/pdf/<id>`.

---

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with HMR on :5173 |
| `npm run build` | Typecheck (`tsc -b`) then production build to `dist/` |
| `npm run preview` | Serve the built bundle locally |
| `npm run lint` | oxlint |

## Troubleshooting

| Symptom | Cause |
|---|---|
| "API unreachable" badge | Backend isn't running — `scripts/run_api.py` in the other repo. |
| `vector_store` badge red | Qdrant is down: `docker start research-rag-qdrant`. |
| `cache` badge amber | Redis is down. Everything still works, just slower: `docker start redis-server`. |
| 401 on every request | Backend has `API_KEYS` set; put a matching key in `VITE_API_KEY` and restart `npm run dev`. |
| CORS error in console | Backend `CORS_ORIGINS` doesn't include this origin. |
| Corpus tab empty | Nothing indexed yet — run `download_corpus.py` then `run_ingestion.py`. |
