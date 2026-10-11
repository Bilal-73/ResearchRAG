# llm/

`LLMClient` — the single wrapper every agent step uses to call Azure OpenAI chat.

- Retries 429 / 5xx / timeouts with exponential backoff + jitter, honouring `Retry-After`.
- A semaphore caps concurrent in-flight calls (`LLM_MAX_CONCURRENCY`).
- `chat_json()` never raises on malformed JSON — it returns a caller-supplied safe default.
- `stats` (calls / retries / rate_limit_hits / failures) feeds the UI sidebar.

Per-*user* request limiting is a different concern and lives in `guardrails/rate_limit.py`.
