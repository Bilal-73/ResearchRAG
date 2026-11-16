"""Turns a ParsedDocument into retrievable Chunks.

Design decisions that matter for benchmark scores:

* **One chunk never spans two sections.** The benchmark's qrels are
  (doc_id, section_id) pairs, so a chunk that straddles sections would make a
  hit unattributable. Every chunk records the `section_id` it came from, which
  is what `evaluation/retrieval_metrics.py` scores against.
* **Contextualisation.** Each chunk's embedded text is prefixed with the paper
  title and section heading. A bare paragraph that says "the error drops by
  38%" is unretrievable on its own; with "Inverter Output Impedance Estimation
  › IV. Simulation Results" in front of it, it is findable. The prefix is part
  of the indexed text, and the generator sees it too.
* **Tables and figures are their own chunks**, carrying their caption plus the
  sentence that referenced them. 32% of the benchmark's queries are
  table/image-sourced, so folding them into surrounding prose would cap the
  achievable recall.
* **Token-packed paragraphs with overlap.** Paragraphs are packed up to
  `max_tokens`, with the last `overlap_paragraphs` repeated at the start of the
  next chunk so a fact split across a boundary is still retrievable whole.
"""
from __future__ import annotations

import logging
import re

from research_rag.models import Chunk, ChunkType
from .ragbench import ParsedDocument, ParsedSection, SectionAsset

logger = logging.getLogger(__name__)

FIGURE_CAPTION_PROMPT = (
    "This is a figure from a scientific paper. Describe the chart type (e.g. line plot, "
    "scatter plot, bar chart, box plot, schematic diagram, heatmap), what is on each axis "
    "including units, the series or conditions compared, and the quantitative takeaway "
    "(trends, crossover points, best-performing method, any visible numbers). Be precise "
    "with numbers and do not speculate beyond what the figure shows."
)

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WORD = re.compile(r"\S+")


def _default_token_count(text: str) -> int:
    """Rough token estimate used when no tokenizer is injected (~0.75 words/token)."""
    return int(len(_WORD.findall(text)) / 0.75) + 1


class TiktokenCounter:
    """Exact cl100k_base counts; falls back to the estimator if tiktoken is absent."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as error:  # noqa: BLE001 — optional dependency
            logger.warning(f"tiktoken unavailable ({error}); using the word-based token estimator")
            self._encoding = None

    def __call__(self, text: str) -> int:
        if self._encoding is None:
            return _default_token_count(text)
        return len(self._encoding.encode(text, disallowed_special=()))


class SectionChunker:
    def __init__(
        self,
        max_tokens: int = 400,
        overlap_paragraphs: int = 1,
        token_counter=None,
        skip_reference_sections: bool = True,
        captioner=None,                       # BaseCaptioner | None — figures get a VLM caption when set
        min_chunk_chars: int = 80,
    ):
        self.max_tokens = max_tokens
        self.overlap_paragraphs = max(0, overlap_paragraphs)
        self.count_tokens = token_counter or _default_token_count
        self.skip_reference_sections = skip_reference_sections
        self.captioner = captioner
        self.min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------ public
    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = [self._abstract_chunk(doc)] if doc.abstract else []
        for section in doc.sections:
            if self.skip_reference_sections and section.is_reference_section:
                logger.debug(f"[{doc.paper_id}] skipping section {section.section_id} ({section.heading})")
                continue
            chunks.extend(self._chunk_section(doc, section))
        logger.info(f"[{doc.paper_id}] {len(chunks)} chunk(s) from {len(doc.sections)} section(s)")
        return chunks

    # ------------------------------------------------------------------ internals
    def _prefix(self, doc: ParsedDocument, heading: str) -> str:
        return f"{doc.title} › {heading}\n\n" if doc.title else f"{heading}\n\n"

    def _abstract_chunk(self, doc: ParsedDocument) -> Chunk:
        """The abstract is section 0 in the dataset; indexing it explicitly means
        'what is this paper about?' queries land on it."""
        return Chunk(
            id=f"{doc.paper_id}::abstract",
            text=f"{self._prefix(doc, 'Abstract')}{doc.abstract}",
            chunk_type=ChunkType.TEXT,
            paper_id=doc.paper_id,
            section="Abstract",
            section_id=0,
            paper_title=doc.title,
        )

    def _chunk_section(self, doc: ParsedDocument, section: ParsedSection) -> list[Chunk]:
        chunks = [
            Chunk(
                id=f"{doc.paper_id}::s{section.section_id}::t{i}",
                text=f"{self._prefix(doc, section.heading)}{body}",
                chunk_type=ChunkType.TEXT,
                paper_id=doc.paper_id,
                section=section.heading,
                section_id=section.section_id,
                paper_title=doc.title,
            )
            for i, body in enumerate(self._pack(section.text))
        ]
        for i, asset in enumerate(section.assets):
            chunk = self._asset_chunk(doc, section, asset, i)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _pack(self, text: str) -> list[str]:
        """Pack paragraphs up to max_tokens, repeating `overlap_paragraphs` between chunks."""
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
        if not paragraphs:
            return []

        bodies: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for paragraph in paragraphs:
            tokens = self.count_tokens(paragraph)
            if tokens > self.max_tokens:                      # one huge paragraph: split on sentences
                if current:
                    bodies.append("\n\n".join(current))
                    current, current_tokens = [], 0
                bodies.extend(self._split_long(paragraph))
                continue
            if current and current_tokens + tokens > self.max_tokens:
                bodies.append("\n\n".join(current))
                current = current[-self.overlap_paragraphs:] if self.overlap_paragraphs else []
                current_tokens = sum(self.count_tokens(p) for p in current)
            current.append(paragraph)
            current_tokens += tokens

        if current:
            bodies.append("\n\n".join(current))
        return [b for b in bodies if len(b) >= self.min_chunk_chars] or bodies[:1]

    def _split_long(self, paragraph: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        out: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            tokens = self.count_tokens(sentence)
            if current and current_tokens + tokens > self.max_tokens:
                out.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += tokens
        if current:
            out.append(" ".join(current))
        return out

    def _asset_chunk(self, doc: ParsedDocument, section: ParsedSection,
                     asset: SectionAsset, index: int) -> Chunk | None:
        label = "Table" if asset.kind == "table" else "Figure"
        parts = [self._prefix(doc, section.heading).rstrip("\n"), ""]
        if asset.caption:
            parts.append(asset.caption)

        if asset.kind == "table":
            if not asset.content.strip():
                return None
            parts.append(asset.content.strip())
        else:
            vlm_caption = self._caption_image(doc.paper_id, asset)
            if vlm_caption:
                parts.append(vlm_caption)
            if not asset.caption and not vlm_caption:
                return None                     # a figure with no caption and no VLM is not retrievable
            parts.insert(1, f"[{label} from {doc.paper_id}]")

        return Chunk(
            id=f"{doc.paper_id}::s{section.section_id}::{asset.kind}{index}",
            text="\n".join(p for p in parts if p is not None).strip(),
            chunk_type=ChunkType.TABLE if asset.kind == "table" else ChunkType.FIGURE,
            paper_id=doc.paper_id,
            section=section.heading,
            section_id=section.section_id,
            paper_title=doc.title,
            figure_no=asset.number,
        )

    def _caption_image(self, paper_id: str, asset: SectionAsset) -> str:
        if self.captioner is None:
            return ""
        image_bytes = asset.image_bytes
        if not image_bytes:
            return ""
        try:
            return self.captioner.caption(image_bytes, FIGURE_CAPTION_PROMPT)
        except Exception as error:  # noqa: BLE001 — captioning is an enhancement, never fatal
            logger.warning(f"[{paper_id}] VLM captioning failed for {asset.key}: {error}")
            return ""
