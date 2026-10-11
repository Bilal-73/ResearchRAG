# vectorstore/

Everything that talks to Qdrant.

- **`base.py`** — `BaseVectorStore` interface: `create_collection`, `upsert`, `hybrid_search`.
- **`qdrant_store.py`** — `QdrantVectorStore`: one collection with a named dense vector
  and a named sparse vector, plus a payload schema (`paper_id`, `section`, `chunk_type`,
  `page_no`, `figure_no`, `image_path`, `text`) that everything downstream (retrieval,
  citations) reads from.

Swapping vector databases means implementing `BaseVectorStore` again — nothing in
`retrieval/` or `generation/` needs to change.
