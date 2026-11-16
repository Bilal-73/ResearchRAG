"""Core data unit: one retrievable piece of a paper (text, table, or figure).

`Chunk` is the domain view used by prompts, guardrails, citations and
evaluation; a LangChain `Document` is the interchange view used by the vector
store, retrievers and rerankers. `to_document()` / `from_document()` convert
losslessly between the two (the payload fields become `Document.metadata`).
"""
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


@dataclass
class Chunk:
    id: str
    text: str
    chunk_type: ChunkType
    paper_id: str                      # arXiv id incl. version, e.g. "2410.14077v2"
    section: str                       # human-readable section heading, e.g. "IV. Simulation Results"
    page_no: int | None = None         # only known for PDF-parsed papers (Docling path)
    section_id: int | None = None      # Open RAGBench section index — what the benchmark's qrels point at
    paper_title: str | None = None

    figure_no: str | None = None
    image_path: str | None = None

    # Set by the reranker at query time. Never stored in Qdrant (not part of the payload).
    score: float | None = None

    def to_payload(self) -> dict:
        """Fields stored as Qdrant payload (vectors are computed by the vector store, never kept here)."""
        return {
            "text": self.text,
            "chunk_type": self.chunk_type.value,
            "paper_id": self.paper_id,
            "section": self.section,
            "page_no": self.page_no,
            "section_id": self.section_id,
            "paper_title": self.paper_title,
            "figure_no": self.figure_no,
            "image_path": self.image_path,
        }

    def to_dict(self) -> dict:
        """JSON-safe form used for LangGraph state (keeps checkpoints serializable
        without registering custom types)."""
        return {"id": self.id, **self.to_payload(), "score": self.score}

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        chunk = cls.from_payload(data["id"], data)
        chunk.score = data.get("score")
        return chunk

    @classmethod
    def from_payload(cls, point_id: str, payload: dict) -> "Chunk":
        return cls(
            id=point_id,
            text=payload["text"],
            chunk_type=ChunkType(payload["chunk_type"]),
            paper_id=payload["paper_id"],
            section=payload.get("section") or "unknown",
            page_no=payload.get("page_no"),
            section_id=payload.get("section_id"),
            paper_title=payload.get("paper_title"),
            figure_no=payload.get("figure_no"),
            image_path=payload.get("image_path"),
        )

    # -- LangChain interchange ------------------------------------------------------
    def to_document(self) -> Document:
        """`page_content` = the text; `metadata` = the rest of the payload, plus the
        reranker score when one has been set."""
        metadata = {k: v for k, v in self.to_payload().items() if k != "text"}
        if self.score is not None:
            metadata["relevance_score"] = self.score
        return Document(id=self.id, page_content=self.text, metadata=metadata)

    @classmethod
    def from_document(cls, doc: Document) -> "Chunk":
        chunk_id = doc.id or doc.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError("Document has no id — cannot map it back to a chunk")
        chunk = cls.from_payload(str(chunk_id), {**doc.metadata, "text": doc.page_content})
        chunk.score = doc.metadata.get("relevance_score")
        return chunk

    def location(self) -> str:
        """Short provenance string used in prompts and citations: '2410.14077v2 › §8 A. Performance comparison'."""
        where = f"§{self.section_id} {self.section}" if self.section_id is not None else self.section
        if self.page_no is not None and self.page_no >= 0:
            where += f", p.{self.page_no}"
        if self.figure_no:
            where += f", {'Table' if self.chunk_type == ChunkType.TABLE else 'Figure'} {self.figure_no}"
        return f"{self.paper_id} › {where}"
