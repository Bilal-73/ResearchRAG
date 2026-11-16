# ingestion/

Turns Open RAG Benchmark corpus files into embedded, stored `Chunk` objects.

The benchmark ships papers **already parsed** into sections, tables and base64 images, so there is
no PDF parsing here at all — no Docling, no OCR, no page layout analysis. That removes the largest
dependency chain in the old pipeline and, more importantly, means every system evaluated against
this benchmark starts from identical source text, so a score difference is a retrieval difference.

- **`ragbench.py`** — `RAGBenchCorpus` reads the mirrored dataset directory;
  `parse_document` / `parse_section` turn raw JSON into `ParsedDocument` / `ParsedSection`.
  This is where the corpus's quirks are handled: tables and figures appear in the section text as
  markdown placeholders (`![table_0](table_0)`) with their caption on the line before or after, so
  the placeholder is removed, the real content substituted, and the caption carried onto the asset.
  Images are base64 data URIs that make a single file hundreds of KB, so they are only decoded when
  `load_images=True`.
- **`section_chunker.py`** — `SectionChunker`: `ParsedDocument` → `Chunk` list.
- **`captioner.py`** — `VLMCaptioner`: figure image → caption text, using a **separate**
  vision-language model (not the embedding model, not the generation LLM). Optional: only used when
  `CAPTION_FIGURES=true`. Defaults to a free, self-hosted Qwen2.5-VL-7B served via Ollama or vLLM
  behind an OpenAI-compatible endpoint — see the serving commands in the docstring. Swapping to a
  hosted API is a `.env` change, not a code change.
- **`pipeline.py`** — `IngestionPipeline`: read → chunk → LangChain `Document`s → `vector_store.add_documents()` (embeds in batches + upserts), writing the
  manifest after every paper so a 400-paper run is resumable.

## Chunking decisions that affect benchmark scores

**One chunk never spans two sections.** The qrels are `(doc_id, section_id)` pairs, so a chunk
straddling a boundary would make a hit unattributable. Every chunk records its `section_id`, which
is what `evaluation/retrieval_metrics.py` scores against.

**Every chunk is contextualised** with the paper title and section heading. A bare paragraph saying
"the error drops by 38%" is unretrievable; prefixed with
`Inverter Output Impedance Estimation › IV. Simulation Results`, it is findable. The prefix is part
of the embedded text and the generator sees it too.

**Tables and figures are their own chunks**, carrying their caption and the referencing sentence.
32% of benchmark queries are table- or image-sourced, so folding them into surrounding prose would
cap achievable recall no matter how good the retriever is.

**Paragraphs are token-packed with overlap** (`CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_PARAGRAPHS`), so a
fact split across a boundary is still retrievable whole. Reference sections are skipped by default —
bibliographies match a lot of queries lexically and answer none of them.

## Why figures become text chunks

Azure OpenAI embeddings are text-only, so rather than maintaining a second (image) vector space in
Qdrant, every figure is reduced to text — the paper's own caption, plus optionally a VLM
description of chart type, axes and visible numbers — and embedded in the *same* space as
everything else. A figure with neither caption nor VLM description is skipped rather than indexed
as an unretrievable stub.
