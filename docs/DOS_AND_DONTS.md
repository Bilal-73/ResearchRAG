# Do's and don'ts

The rules this codebase follows. Most of them exist because breaking them once caused a real bug.

## Secrets and configuration

| ✅ Do | ❌ Don't |
|---|---|
| Keep keys in `backend/.env` and `frontend/.env.local`. Both are git-ignored. | Commit `.env`, `.env.local`, `.env.bak` or any file containing a key, even "temporarily". |
| Add every new setting to `backend/config/settings.py` **and** `backend/.env.example` (with a comment). | Read `os.environ` anywhere else. Settings are injected through constructors. |
| Use placeholder values in `.env.example` (`<your-key>`, `<your-resource>`). | Put real resource names, endpoints or tenant ids in examples or docs. |
| Rotate a key immediately if it was ever pushed, even after deleting it. Git history keeps it. | Rely on deleting a file to remove a leaked secret. |
| Treat `VITE_API_KEY` as public. | Put a production secret in any `VITE_*` variable. |

## Running the system

| ✅ Do | ❌ Don't |
|---|---|
| Run backend scripts **from `backend/`** with the virtualenv's Python. | Run them from the repo root, or with the system `python`. Paths and imports will break. |
| Start small: `download_corpus.py --limit 50`. | Ingest 1,000 papers before checking that 50 work. |
| Check `/health` before debugging anything else. | Assume a bad answer is a retrieval bug before checking that the paper is indexed (the Corpus tab). |
| Set `API_KEYS` and a specific `CORS_ORIGINS` before binding to `0.0.0.0`. | Expose an API with no auth. `/health` will report `auth: degraded`. |
| Use `CHECKPOINT_BACKEND=sqlite` with more than one worker. | Run multiple workers on in-memory checkpoints. Follow-ups will lose their history. |
| Leave live RAGAS scoring off by default. It adds about 15–20 s per turn. | Turn on scoring for every request in a demo or in production. |

## Backend code

| ✅ Do | ❌ Don't |
|---|---|
| Choose implementations in `bootstrap.py`, the composition root. | Instantiate concrete classes (Qdrant, Azure, Redis) deep inside other modules. |
| Use LangChain interfaces (`Embeddings`, `VectorStore`, `BaseRetriever`, `Document`) between layers. | Pass raw SDK objects (`openai`, `qdrant_client`) across module boundaries. |
| Return **200** for refusals, redirects and safety blocks, and set `answer_kind`. | Turn a correct refusal into a 4xx or 5xx. |
| Return **503** when a dependency is down. | Fabricate an answer when the vector store or LLM is unreachable. |
| Keep every chunk inside one section and preserve `section_id`. | Chunk across section boundaries. Section-level metrics depend on it. |
| Register a new guardrail in `guardrails/registry.py` with its setting flag. | Add a check the UI's Guardrails tab cannot see. |
| Bound every loop in the agent graph (`MAX_RETRIEVAL_RETRIES`). | Add a conditional edge that can cycle forever. |
| Store only PII-redacted text in memory and logs. | Log raw user messages or API keys. |
| Fail open on the cache. Redis is an optimisation. | Make a request fail because Redis is down. |
| Add or update a test for every behaviour change. The suite is offline and runs in seconds. | Write tests that call Azure, Qdrant or the network. Use the fakes in `tests/fakes.py`. |
| Update the module's `README.md` when you change what it does. | Let a folder README describe code that no longer exists. |

## Frontend code

| ✅ Do | ❌ Don't |
|---|---|
| Keep all HTTP calls in `src/api/client.ts`. | Call `fetch` from components. |
| Mirror every backend schema change in `src/api/types.ts` in the **same commit**. | Let the TypeScript types drift from `/openapi.json`. |
| Render answer and paper text as text nodes. | Use `dangerouslySetInnerHTML`. Answers contain untrusted retrieved content. |
| Branch on `answer_kind` and `blocked` for refusals, and show them as normal messages with a badge. | Show a correct refusal as an error card. |
| Show `—` for a metric that could not be computed. | Show `0` for a missing score. |
| Run `npm run lint` and `npm run build` before pushing. | Push code that fails `tsc -b`. |

## Evaluation

| ✅ Do | ❌ Don't |
|---|---|
| Use a fixed seed and stratified sampling so runs can be compared. | Compare two runs with different sample sizes or seeds and draw conclusions. |
| Score only queries whose gold paper is indexed, and report how many were dropped. | Count un-indexed papers as retrieval failures. |
| Report section-level metrics alongside document-level ones. | Report only the friendlier document-level number. |
| Run `--judge builtin` as a second opinion. The RAGAS judge shares a model family with the generator. | Treat a single LLM-judge score as ground truth. |
| Commit only small result summaries under `backend/data/eval/`. | Commit the downloaded corpus, Qdrant data or checkpoints. |

## Git and collaboration

| ✅ Do | ❌ Don't |
|---|---|
| Branch from `main` (`feature/…`, `fix/…`, `docs/…`) and open a pull request. | Push directly to `main`. |
| Write commit messages that say **why**, in the imperative ("Add …", "Fix …"). | Write "update", "wip" or "changes". |
| Keep a pull request to one concern: backend, frontend, or a contract change that touches both. | Mix a refactor, a feature and a formatting pass in one PR. |
| Wait for CI to pass before merging. | Merge with failing tests or lint. |
