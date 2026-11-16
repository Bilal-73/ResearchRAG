# agents/

The LangGraph agent. `graph.py` wires it, `nodes.py` holds every step, `state.py` defines the state.
Each step that calls the LLM runs an LCEL chain (`ChatPromptTemplate` from `prompts.py` `|` the
`LLMClient` runnable `|` a parser), and retrieval steps pass LangChain `Document`s between the
retriever and reranker before storing them in graph state as plain chunk dicts.

- **`intent.py`** — classifier: follow_up / new_question / chit_chat / out_of_scope. Rule → LLM →
  safe fallbacks. Note the asymmetry in the guards: low confidence collapses to `new_question`,
  because routing a real question to grounded retrieval is harmless while mis-routing it to
  chit-chat loses the answer.
- **`planner.py`** — splits a multi-part question into self-contained sub-questions, retrieved
  separately. A cheap regex decides whether to ask the LLM at all, so simple questions pay nothing.
- **`memory.py`** — bounded history + rolling LLM summary; only PII-masked text is stored.
- **`responses.py`** — deterministic replies (out of scope, not found, upstream errors) so wording
  that matters never depends on an LLM being available or cooperative.
- **`routing.py`** — explains a finished turn: which agents ran, in what order, and why. Derived
  from the compiled graph and the real trace, so it cannot drift from the actual wiring. This is
  what `diagnostics.route` / `diagnostics.explanation` return over the API.
- **`graph.py`** — `ResearchRAGAgent.chat()` / `chat_stream()`, checkpointer factory
  (`memory` | `sqlite`), per-conversation memory.

Two details worth knowing:

- **`session_id` is the API client, not the conversation.** Rate limiting keys off it, so a caller
  can't reset its quota by starting a new conversation each request. Memory keys off `thread_id`.
- **The corrective-retrieval loop is capped** by `MAX_RETRIEVAL_RETRIES`, with a LangGraph
  `recursion_limit` as a second backstop, so the graph can never spin and burn quota.
