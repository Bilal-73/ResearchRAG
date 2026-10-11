"""Core data unit: one retrievable piece of a paper (text, table, or figure)."""
from dataclasses import dataclass
from enum import Enum


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

    dense_vector: list[float] | None = None
    sparse_vector: dict[str, list] | None = None  # {"indices": [...], "values": [...]}

    # Set by the reranker at query time. Never stored in Qdrant (not part of the payload).
    score: float | None = None

    def to_payload(self) -> dict:
        """Fields stored as Qdrant payload (everything except the vectors themselves)."""
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
        without registering custom types). Vectors are deliberately dropped."""
        return {"id": self.id, **self.to_payload(), "score": self.score}

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        chunk = cls.from_payload(data["id"], data)
        chunk.score = data.get("score")
        return chunk

    @classmethod
    def from_payload(cls, point_id: str, payload: dict, dense_vector: list[float] | None = None) -> "Chunk":
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
            dense_vector=dense_vector,
        )

    def location(self) -> str:
        """Short provenance string used in prompts and citations: '2410.14077v2 › §8 A. Performance comparison'."""
        where = f"§{self.section_id} {self.section}" if self.section_id is not None else self.section
        if self.page_no is not None and self.page_no >= 0:
            where += f", p.{self.page_no}"
        if self.figure_no:
            where += f", {'Table' if self.chunk_type == ChunkType.TABLE else 'Figure'} {self.figure_no}"
        return f"{self.paper_id} › {where}"
