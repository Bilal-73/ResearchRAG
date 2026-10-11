# retrieval/

Query -> ranked chunks.

- **`base.py`** — `BaseRetriever`, `BaseReranker` interfaces.
- **`hybrid_retriever.py`** — `HybridRetriever`: embeds the query (dense + sparse) and
  calls the vector store's hybrid search. **No paper_id filter comes from the caller** —
  relevance is decided entirely by the system, per the single-paper-focus requirement.
- **`reranker.py`** — `CrossEncoderReranker`: reorders the top candidates for precision
  before they reach generation.

This is the stage to benchmark with `evaluation/retrieval_metrics.py` — swap
`HybridRetriever` for a dense-only version and compare Recall@k/MRR to see whether hybrid
search is actually earning its complexity on this corpus.

## Advanced retrieval (added)
- **`query_transforms.py`** — `QueryRewriter` (standalone/expanded/broadened queries) and `HyDEGenerator`.
- **`fusion.py`** — Reciprocal Rank Fusion across query variants.
- **`advanced_retriever.py`** — `AdvancedRetriever`: multi-query hybrid search + RRF; also satisfies `BaseRetriever` so evaluation uses the same retriever as the chatbot.
- **`reranker.py`** — cross-encoder now writes `chunk.score`; `NoOpReranker` when reranking is off.
