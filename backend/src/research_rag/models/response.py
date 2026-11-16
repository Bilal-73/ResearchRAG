"""What the generator returns — an answer that is always paired with citations."""
import re
from dataclasses import dataclass, field

from .chunk import Chunk

_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def arxiv_url(paper_id: str) -> str | None:
    """Abstract-page URL for an arXiv id; None for non-arXiv papers (e.g. local PDFs)."""
    return f"https://arxiv.org/abs/{paper_id}" if _ARXIV_ID.match(paper_id) else None


def arxiv_pdf_url(paper_id: str) -> str | None:
    """Direct PDF link. The benchmark corpus ships pre-parsed JSON and stores no PDFs,
    so this is the only way to reach the original document."""
    return f"https://arxiv.org/pdf/{paper_id}" if _ARXIV_ID.match(paper_id) else None


@dataclass
class SourceCitation:
    paper_id: str
    section: str
    page_no: int | None
    snippet: str
    figure_no: str | None = None

    # ---- attribution details (all optional so older call sites keep working) ----
    label: str = ""                   # "S1", "S2" ... — the marker used inline in the answer as [S1]
    chunk_type: str = "text"          # text | table | figure
    score: float | None = None        # reranker relevance score, when available
    paper_title: str | None = None
    cited: bool = False               # True if the answer actually references this source inline
    section_id: int | None = None
    url: str | None = None            # arXiv abstract page

    def format(self) -> str:
        where = f"§{self.section_id} {self.section}" if self.section_id is not None else self.section
        base = f"{self.paper_id} › {where}"
        if self.page_no is not None and self.page_no >= 0:
            base += f", p.{self.page_no}"
        return f"{base}, Figure {self.figure_no}" if self.figure_no else base

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id, "section": self.section, "page_no": self.page_no,
            "snippet": self.snippet, "figure_no": self.figure_no, "label": self.label,
            "chunk_type": self.chunk_type, "score": self.score,
            "paper_title": self.paper_title, "cited": self.cited,
            "section_id": self.section_id, "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceCitation":
        return cls(**data)


@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceCitation]                  # built by the system, never by the caller
    retrieved_chunks: list[Chunk] = field(default_factory=list)
