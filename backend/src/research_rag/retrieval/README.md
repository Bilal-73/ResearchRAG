# retrieval/

Query -> ranked LangChain `Document`s.

- **`hybrid_retriever.py`** — `HybridRetriever`, a LangChain `BaseRetriever` over the vector store's
  hybrid (dense + sparse) search. **No paper_id filter comes from the caller** — relevance is decided
  entirely by the system, per the single-paper-focus requirement.
- **`advanced_retriever.py`** — `AdvancedRetriever`, also a `BaseRetriever`: multi-query hybrid search
  (original + rewritten + alternatives + HyDE) fused with RRF. `invoke(query)` runs the whole technique
  end to end, so evaluation uses the same retriever as the chatbot; the agent calls `search_variants()`
  so each step shows up separately in its trace.
- **`query_transforms.py`** — `QueryRewriter` (standalone/expanded/broadened queries) and `HyDEGenerator`,
  both LCEL chains (`ChatPromptTemplate | llm | parser`).
- **`fusion.py`** — Reciprocal Rank Fusion across query variants (matches items by `.id`).
- **`reranker.py`** — `CrossEncoderReranker`, a LangChain `BaseDocumentCompressor` that writes
  `metadata["relevance_score"]`; `NoOpReranker` when reranking is off.

This is the stage to benchmark with `evaluation/retrieval_metrics.py` — swap `AdvancedRetriever` for
`HybridRetriever` and compare Recall@k/MRR to see whether multi-query search is earning its cost on
this corpus.
