# embeddings/

Text -> vector, and query -> sparse vector, in one place.

- **`base.py`** — `BaseEmbedder` interface: `embed`, `embed_batch`, `sparse_encode`.
- **`azure_embedder.py`** — `AzureEmbedder`: dense vectors via Azure OpenAI, plus a
  lightweight hashing-based sparse vector for hybrid search (swap for SPLADE/BM25-over-corpus
  later without touching any other module).

Every chunk (text, table, or figure caption) goes through the same embedder — there is
no separate image embedding path. See `ingestion/README.md` for why.
