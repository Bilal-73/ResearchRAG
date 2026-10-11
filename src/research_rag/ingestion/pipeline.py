"""Orchestrates the corpus through read -> chunk -> embed -> upsert.

Resumable by design: ingesting 400 papers is a long job, so progress is
written to a state file after every paper. Re-running skips what is already
indexed unless `force=True`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from research_rag.embeddings.base import BaseEmbedder
from research_rag.models import Chunk
from research_rag.vectorstore.base import BaseVectorStore
from .ragbench import RAGBenchCorpus
from .section_chunker import SectionChunker

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64          # Azure accepts large batches; this keeps each request well under limits


@dataclass
class IngestionReport:
    papers_ingested: int = 0
    papers_skipped: int = 0
    papers_failed: int = 0
    chunks_upserted: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "papers_ingested": self.papers_ingested, "papers_skipped": self.papers_skipped,
            "papers_failed": self.papers_failed, "chunks_upserted": self.chunks_upserted,
            "by_type": self.by_type, "failures": self.failures, "seconds": round(self.seconds, 1),
        }


class IngestionPipeline:
    def __init__(
        self,
        corpus: RAGBenchCorpus,
        chunker: SectionChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        manifest_path: str | Path = "data/arxiv_manifest.json",
        load_images: bool = False,
    ):
        self.corpus = corpus
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.manifest_path = Path(manifest_path)
        self.load_images = load_images

    # ------------------------------------------------------------------ manifest
    def _read_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(f"Manifest {self.manifest_path} is corrupt — starting a new one")
        return {}

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.manifest_path)       # atomic: a crash mid-write can't corrupt the manifest

    # ------------------------------------------------------------------ ingestion
    def ingest_paper(self, paper_id: str) -> list[Chunk]:
        doc = self.corpus.document(paper_id, load_images=self.load_images)
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            logger.warning(f"[{paper_id}] produced no chunks — nothing to index")
            return []

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            vectors = self.embedder.embed_batch([c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                chunk.dense_vector = vector
                chunk.sparse_vector = self.embedder.sparse_encode(chunk.text)
            self.vector_store.upsert(batch)

        return chunks

    def ingest(self, paper_ids: list[str] | None = None, force: bool = False,
               progress=None) -> IngestionReport:
        """`progress` is an optional callable(done, total, paper_id) for CLI output."""
        ids = paper_ids if paper_ids is not None else self.corpus.available_ids()
        manifest = self._read_manifest()
        report = IngestionReport()
        t_start = time.monotonic()

        for i, paper_id in enumerate(ids, start=1):
            if not force and paper_id in manifest and manifest[paper_id].get("n_chunks"):
                report.papers_skipped += 1
                if progress:
                    progress(i, len(ids), f"{paper_id} (already indexed)")
                continue
            try:
                chunks = self.ingest_paper(paper_id)
            except Exception as error:  # noqa: BLE001 — one bad paper must not sink a 400-paper run
                logger.error(f"[{paper_id}] ingestion failed: {error}")
                report.papers_failed += 1
                report.failures[paper_id] = str(error)
                if progress:
                    progress(i, len(ids), f"{paper_id} FAILED: {error}")
                continue

            doc = self.corpus.document(paper_id)
            manifest[paper_id] = doc.manifest_entry(n_chunks=len(chunks))
            self._write_manifest(manifest)

            report.papers_ingested += 1
            report.chunks_upserted += len(chunks)
            for chunk in chunks:
                report.by_type[chunk.chunk_type.value] = report.by_type.get(chunk.chunk_type.value, 0) + 1
            if progress:
                progress(i, len(ids), f"{paper_id} ({len(chunks)} chunks)")

        report.seconds = time.monotonic() - t_start
        logger.info(f"Ingestion complete: {report.to_dict()}")
        return report
