"""Reader for the Open RAG Benchmark corpus (vectara/open_ragbench, arXiv split).

Layout mirrored under ``data/open_ragbench/`` (see scripts/download_corpus.py):

    queries.json   {query_uuid: {"query": str, "type": "abstractive"|"extractive",
                                 "source": "text"|"text-image"|"text-table"|"text-table-image"}}
    qrels.json     {query_uuid: {"doc_id": str, "section_id": int}}
    answers.json   {query_uuid: str}
    pdf_urls.json  {doc_id: url}
    corpus/{doc_id}.json
        {"id", "title", "authors": [...], "categories": [...], "abstract",
         "published", "updated",
         "sections": [{"section_id": int, "text": str,
                       "tables": {"table_0": "<markdown table>"},
                       "images": {"img-0.jpeg": "data:image/jpeg;base64,..."}}]}

Two things about the section text matter for chunking, and both are handled in
`parse_section`:

* Tables and images appear in the text as markdown placeholders —
  ``![table_0](table_0)`` / ``![img-0.jpeg](img-0.jpeg)`` — with their caption
  usually on the line immediately before and discussion after. The placeholder
  is useless to a retriever, but the caption is exactly what makes a figure
  findable, so each placeholder is replaced by its real content and the caption
  is carried onto the table/figure chunk.
* The heading sits at the top of the section text as markdown ("## IV. Results").
  It is the only section label the dataset gives us, so it becomes `Chunk.section`.

Images are base64 data URIs embedded in the JSON — a single corpus file can be
hundreds of KB because of them. `iter_documents` therefore has a
``load_images`` switch, off by default, so ingestion that doesn't caption
figures never pays to decode them.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<key>[^)]+)\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*(?P<title>.+?)\s*#*\s*$", re.MULTILINE)
_FOOTNOTE_MARKER = re.compile(r"\[\^\d+\]")
# "Fig. 3:", "Figure 3.", "Table I:", "TABLE 2 -" ... at the start of a line
_CAPTION_LINE = re.compile(r"^\s*(fig(?:ure)?\.?|table|tab\.)\s*([IVXLC\d]+[a-z]?)\s*[.:\-–—]?\s+\S",
                           re.IGNORECASE)
_REFERENCE_HEADING = re.compile(r"^\s*(references?|bibliography|acknowledg(e)?ments?)\s*$", re.IGNORECASE)


@dataclass
class SectionAsset:
    """A table or figure found inside a section."""
    key: str                  # "table_0" / "img-0.jpeg"
    kind: str                 # "table" | "figure"
    caption: str = ""         # caption line found next to the placeholder
    content: str = ""         # markdown table source (tables only)
    data_uri: str = ""        # base64 image (figures only, when load_images=True)
    number: str | None = None  # "1", "III" — parsed out of the caption when present

    @property
    def image_bytes(self) -> bytes | None:
        if not self.data_uri or "," not in self.data_uri:
            return None
        try:
            return base64.b64decode(self.data_uri.split(",", 1)[1])
        except (ValueError, TypeError) as error:
            logger.warning(f"Could not decode image {self.key}: {error}")
            return None


@dataclass
class ParsedSection:
    section_id: int
    heading: str
    text: str                                   # prose with placeholders removed
    assets: list[SectionAsset] = field(default_factory=list)

    @property
    def is_reference_section(self) -> bool:
        return bool(_REFERENCE_HEADING.match(self.heading.strip()))


@dataclass
class ParsedDocument:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str | None
    updated: str | None
    sections: list[ParsedSection]

    def manifest_entry(self, n_chunks: int | None = None) -> dict:
        return {"title": self.title, "authors": self.authors, "categories": self.categories,
                "published": self.published, "updated": self.updated, "n_chunks": n_chunks}


def _clean_heading(raw_text: str, section_id: int) -> str:
    """The section heading is the first markdown heading in the section body."""
    match = _HEADING.search(raw_text[:400])
    if not match:
        first_line = raw_text.strip().splitlines()[0] if raw_text.strip() else ""
        return (first_line[:80].strip() or f"section {section_id}")
    return " ".join(match.group("title").split())[:120]


def _caption_near(lines: list[str], index: int) -> str:
    """Caption for a placeholder on line `index`: the nearest caption-looking line
    just before it, else just after (papers do both)."""
    for offset in (1, 2, 3):
        if index - offset >= 0:
            candidate = lines[index - offset].strip()
            if candidate and _CAPTION_LINE.match(candidate):
                return candidate
    for offset in (1, 2):
        if index + offset < len(lines):
            candidate = lines[index + offset].strip()
            if candidate and _CAPTION_LINE.match(candidate):
                return candidate
    return ""


def _caption_number(caption: str) -> str | None:
    match = _CAPTION_LINE.match(caption)
    return match.group(2) if match else None


def parse_section(raw: dict, load_images: bool = False) -> ParsedSection:
    """Split one raw section into prose + its tables/figures."""
    section_id = int(raw.get("section_id", 0))
    text = raw.get("text") or ""
    tables = raw.get("tables") or {}
    images = raw.get("images") or {}

    lines = text.splitlines()
    assets: list[SectionAsset] = []
    kept: list[str] = []

    for i, line in enumerate(lines):
        match = _PLACEHOLDER.search(line)
        if not match:
            kept.append(line)
            continue
        key = match.group("key").strip()
        caption = _caption_near(lines, i)
        if key in tables:
            assets.append(SectionAsset(key=key, kind="table", caption=caption,
                                       content=str(tables[key]), number=_caption_number(caption)))
        elif key in images:
            assets.append(SectionAsset(key=key, kind="figure", caption=caption,
                                       data_uri=str(images[key]) if load_images else "",
                                       number=_caption_number(caption)))
        # Drop the placeholder line itself; keep anything else that shared the line.
        remainder = _PLACEHOLDER.sub("", line).strip()
        if remainder:
            kept.append(remainder)

    # Assets whose placeholder never appeared in the text (it happens) are still indexed.
    seen = {a.key for a in assets}
    for key, content in tables.items():
        if key not in seen:
            assets.append(SectionAsset(key=key, kind="table", content=str(content)))
    for key, data_uri in images.items():
        if key not in seen:
            assets.append(SectionAsset(key=key, kind="figure",
                                       data_uri=str(data_uri) if load_images else ""))

    prose = _FOOTNOTE_MARKER.sub("", "\n".join(kept))
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
    return ParsedSection(section_id=section_id, heading=_clean_heading(text, section_id),
                         text=prose, assets=assets)


def parse_document(raw: dict, load_images: bool = False) -> ParsedDocument:
    return ParsedDocument(
        paper_id=str(raw.get("id") or "").strip(),
        title=" ".join(str(raw.get("title") or "").split()),
        abstract=" ".join(str(raw.get("abstract") or "").split()),
        authors=[str(a) for a in (raw.get("authors") or [])],
        categories=[str(c) for c in (raw.get("categories") or [])],
        published=raw.get("published"),
        updated=raw.get("updated"),
        sections=[parse_section(s, load_images) for s in (raw.get("sections") or [])],
    )


class RAGBenchCorpus:
    """Reads the mirrored dataset directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.corpus_dir = self.root / "corpus"

    # -- metadata files ------------------------------------------------------------
    def _load(self, name: str) -> dict:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/download_corpus.py --split arxiv"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def queries(self) -> dict[str, dict]:
        return self._load("queries.json")

    def qrels(self) -> dict[str, dict]:
        return self._load("qrels.json")

    def answers(self) -> dict[str, str]:
        return self._load("answers.json")

    def pdf_urls(self) -> dict[str, str]:
        return self._load("pdf_urls.json")

    # -- corpus documents ----------------------------------------------------------
    def available_ids(self) -> list[str]:
        if not self.corpus_dir.exists():
            return []
        return sorted(p.stem for p in self.corpus_dir.glob("*.json"))

    def judged_ids(self) -> list[str]:
        """Papers that at least one query is judged against — the 'positive' documents.
        Ingesting these alone gives a benchmark-complete corpus at a fraction of the cost."""
        return sorted({entry["doc_id"] for entry in self.qrels().values()})

    def document(self, paper_id: str, load_images: bool = False) -> ParsedDocument:
        path = self.corpus_dir / f"{paper_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Corpus file missing for {paper_id}: {path}")
        return parse_document(json.loads(path.read_text(encoding="utf-8")), load_images)

    def iter_documents(self, paper_ids: list[str] | None = None,
                       load_images: bool = False) -> Iterator[ParsedDocument]:
        for paper_id in (paper_ids if paper_ids is not None else self.available_ids()):
            try:
                yield self.document(paper_id, load_images)
            except (FileNotFoundError, json.JSONDecodeError) as error:
                logger.error(f"Skipping {paper_id}: {error}")
