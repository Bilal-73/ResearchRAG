# vectorstore/

Everything that talks to Qdrant.

- **`qdrant_store.py`** — `QdrantHybridVectorStore`, a LangChain `VectorStore`: one collection with a
  named dense vector and a named sparse vector, plus a flat payload schema (`paper_id`, `section`,
  `section_id`, `chunk_type`, `page_no`, `figure_no`, `image_path`, `paper_title`, `text`) that
  everything downstream (retrieval, citations) reads from.

It owns its encoders — a LangChain `Embeddings` for dense vectors and a `HashingSparseEncoder` for
sparse ones — so ingestion is `store.add_documents(docs)` and search is `store.similarity_search(q)`
(dense + sparse, fused server-side with RRF). `hybrid_search_by_vector()` takes precomputed vectors so
the multi-query retriever can embed every query variant in one API call.

Swapping vector databases means writing another LangChain `VectorStore` with the same three extras
(`hybrid_search_by_vector`, `list_paper_ids`, `count_chunks`) — nothing in `retrieval/` or
`generation/` needs to change.
