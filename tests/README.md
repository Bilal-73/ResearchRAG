# tests/

Fully offline. Azure and Qdrant are faked; everything else is real — the actual LangGraph graph,
the actual guardrails, the actual FastAPI app. That is what the abstract base classes in each
module buy you.

- **`fakes.py`** — the shared doubles. `FakeChatModel` routes by system-prompt marker and each
  response is overridable per test, so you can script a grader outage, a hallucinated citation or
  an uncited answer without touching the code under test. `Harness` wires a complete agent.
- **`test_agent_graph.py`** — end-to-end agent behaviour: routing, memory, follow-ups, corrective
  retrieval, citation enforcement, every guardrail path.
- **`test_api.py`** — the HTTP contract: status codes, the uniform error body, auth, rate-limit
  headers, OpenAPI completeness. Injects `app.state.services` directly to bypass `lifespan`.
- **`test_ingestion.py`** — parsing and chunking against a miniature Open RAG Benchmark corpus
  that reproduces the real dataset's quirks (placeholders, captions, base64 images, references).
- **`test_evaluation.py`** — the harness itself: that the eval set excludes unindexed gold
  documents, that sampling is stratified and reproducible, and that section-level scoring really
  is stricter than document-level.
- **`test_guardrails.py`** — each guardrail in isolation, including the false-positive cases
  (scientific notation must not be masked as PII; "ignore prior context" is not an injection;
  discussing hazardous research is not a dual-use request).
- **`test_routing.py`**, **`test_planner.py`**, **`test_retrieval.py`**, **`test_retriever.py`**,
  **`test_llm_client.py`**, **`test_chunker.py`**, **`test_rag_system.py`** — per-component.

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```
