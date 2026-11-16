# llm/

The LangChain model layer.

- **`factory.py`** — the one place model objects are built: `build_chat_model()` (`AzureChatOpenAI`),
  `build_embeddings()` (`AzureOpenAIEmbeddings`), `build_vision_model()` (`ChatOpenAI` against any
  OpenAI-compatible vision endpoint). The agent, ingestion, RAGAS and the captioner all use these.
- **`client.py`** — `LLMClient`, the single wrapper every LLM step goes through. It holds a
  LangChain `BaseChatModel` and builds the steps' LCEL chains:
  - `text_chain(prompt, purpose=...)` → `prompt | llm` → `str`
  - `json_chain(prompt, purpose=..., default=...)` → `prompt | llm(json mode) | JsonObjectOutputParser`,
    with `.with_fallbacks()` returning `default` on an unreachable model or junk output, so a flaky
    classifier can't take the graph down.
  - `chat()` / `chat_json()` for one-off message lists.
- **`parsers.py`** — `JsonObjectOutputParser` (tolerant of code fences and surrounding prose).

Resilience is owned here, not by the SDK (the model is built with `max_retries=0`):

- Retries 429 / 5xx / timeouts with exponential backoff + jitter, honouring `Retry-After`.
- A semaphore caps concurrent in-flight calls (`LLM_MAX_CONCURRENCY`).
- `stats` (calls / retries / rate_limit_hits / failures) feeds `/health`.
- Every call runs with `run_name=<purpose>`, so LangSmith/callback traces name each step.

Per-*user* request limiting is a different concern and lives in `guardrails/rate_limit.py`.
