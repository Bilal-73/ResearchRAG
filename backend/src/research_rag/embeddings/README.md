# embeddings/

Text -> dense vector (LangChain `Embeddings`) and text -> sparse vector, in one place.

- **`cached.py`** — `CachedEmbeddings`: a LangChain `Embeddings` that wraps any embedding model
  (`AzureOpenAIEmbeddings`, built in `llm/factory.py`) with a Redis cache, so a repeated text costs
  no API call.
- **`sparse.py`** — `HashingSparseEncoder`: a lightweight hashing-based sparse vector for the lexical
  half of hybrid search. LangChain has no core sparse interface, so the vector store holds this next to
  its dense `Embeddings` (swap for SPLADE/BM25-over-corpus later without touching any other module).

Both are owned by the vector store (`vectorstore/qdrant_store.py`): it embeds documents on
`add_documents()` and queries on `similarity_search()`. Every chunk (text, table, or figure caption)
goes through the same embeddings — there is no separate image embedding path. See
`ingestion/README.md` for why.
