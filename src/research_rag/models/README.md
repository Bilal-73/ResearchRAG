# models/

Plain data schemas — no logic, no I/O. Every other module passes these objects around.

- **`chunk.py`** — `Chunk` and `ChunkType`: one retrievable unit (text/table/figure) plus its
  metadata (`paper_id`, `section`, **`section_id`**, `paper_title`, `page_no`, vectors).
  `section_id` is the Open RAG Benchmark section index — the thing its qrels point at — so it has
  to survive serialisation into LangGraph state and back, which `test_chunker.py` pins down.
  `location()` renders the provenance string used in prompts and citations.
- **`response.py`** — `SourceCitation` and `RAGResponse`. `sources` is always populated by the
  system from chunk metadata; there is no path where a caller supplies or overrides a citation.
  `arxiv_url()` builds the abstract-page link, returning `None` for non-arXiv ids rather than
  fabricating a URL.

These are deliberately plain dataclasses rather than Pydantic models: they cross the LangGraph
checkpoint boundary as JSON, and the API's Pydantic schemas (`api/schemas.py`) are kept separate
so the wire contract can evolve without changing internal structures.
