# Roadmap

What exists today, its known limitations, and improvements that fit the current architecture. Each item
names the place in the code where it would land.

## Known limitations

| Limitation | Impact | Where a fix lands |
|---|---|---|
| Sparse vectors use a **hashing encoder**, not BM25 or SPLADE | Lexical matching is approximate | `embeddings/sparse.py` (`HashingSparseEncoder.encode`) |
| Figure captions default to the paper's own caption text | Text-image recall is lower than it could be | `CAPTION_FIGURES=true` with a VLM. Measure before paying for it at scale. |
| Conversation memory is per process unless SQLite is used | Multi-worker deployments lose follow-up context | `CHECKPOINT_BACKEND=sqlite` today. A Redis or Postgres checkpointer would be next. |
| The LLM judge shares a model family with the generator | Possible self-preference bias in RAGAS scores | `evaluation/`: add a judge from a different provider |
| Rate limiting is in memory | Limits are per process, not per deployment | `guardrails/rate_limit.py`: back it with Redis |
| Frontend types are mirrored by hand | Schema drift is caught only at review time | Generate `types.ts` from `/openapi.json` |
| `VITE_API_KEY` is visible in the bundle | Not suitable for production auth | A backend-for-frontend proxy with session auth |

## Improvements that can be implemented

### Retrieval and answer quality
- [ ] Real BM25 or SPLADE sparse vectors in place of the hashing encoder
- [ ] Cache query rewrites and HyDE drafts, so repeated questions reach a higher cache hit rate
- [ ] Metadata filters in the API (arXiv category, year, paper id) using Qdrant payload filters
- [ ] Parent-document retrieval: return the whole section around a matching chunk
- [ ] Streaming answers over Server-Sent Events. The agent already exposes `chat_stream()`.

### Evaluation
- [ ] A CI job that runs a small, fixed retrieval-only evaluation and fails on regression
- [ ] A second-provider judge to cross-check RAGAS
- [ ] Track results over time in a small dashboard built from `data/eval/*.json`

### Platform and operations
- [ ] Dockerfiles for `backend/` and `frontend/`, and a full-stack `docker compose` profile
- [ ] Structured JSON logs and OpenTelemetry traces keyed by `X-Request-ID`
- [ ] Redis-backed rate limiting and checkpointing for horizontal scaling
- [ ] Generate `frontend/src/api/types.ts` from the OpenAPI spec in CI
- [ ] Dependabot for pip and npm, and CodeQL scanning
- [ ] Pre-commit hooks: ruff, oxlint, and a secret scanner

### Frontend
- [ ] Frontend unit tests (Vitest + Testing Library)
- [ ] Show streaming tokens as they arrive
- [ ] Persist conversations in the browser between reloads
- [ ] Export a conversation with its citations as Markdown

Contributions toward any of these are welcome. See [CONTRIBUTING.md](../CONTRIBUTING.md).
