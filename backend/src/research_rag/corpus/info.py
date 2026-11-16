"""Live picture of what the assistant can actually answer from.

Which papers are *searchable* is asked of the vector store (cached briefly),
never assumed. The manifest written by ingestion adds titles and arXiv
categories.

Unlike a hand-curated corpus, this one is large (hundreds of arXiv papers
across every category), so `describe()` cannot list it paper by paper: it
summarises by category with a few example titles, which is what the classifier
and the scope-redirect actually need in order to judge "could this corpus
plausibly cover the question?".
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Primary arXiv category prefixes -> readable field names, for the prompt text.
FIELD_NAMES = {
    "cs": "computer science", "math": "mathematics", "stat": "statistics",
    "physics": "physics", "astro-ph": "astrophysics", "cond-mat": "condensed matter physics",
    "hep": "high-energy physics", "gr-qc": "general relativity & quantum cosmology",
    "quant-ph": "quantum physics", "nucl": "nuclear physics", "q-bio": "quantitative biology",
    "q-fin": "quantitative finance", "econ": "economics", "eess": "electrical engineering & systems science",
    "nlin": "nonlinear sciences",
}


def field_of(category: str) -> str:
    """'cs.LG' -> 'computer science'; 'hep-th' -> 'high-energy physics'."""
    head = category.split(".", 1)[0]
    return FIELD_NAMES.get(head, FIELD_NAMES.get(head.split("-", 1)[0], head))


class CorpusInfo:
    def __init__(self, vector_store, manifest_path: str | Path | None = None, ttl_seconds: float = 60.0):
        self._store = vector_store
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._cached_ids: list[str] = []
        self._cached_at: float = 0.0
        self._error: str | None = None
        self._manifest: dict[str, dict] = {}
        self._manifest_path = Path(manifest_path) if manifest_path else None
        self.reload_manifest()

    # -- manifest ------------------------------------------------------------------
    def reload_manifest(self) -> None:
        """Re-read the manifest from disk (ingestion rewrites it as papers are added)."""
        if self._manifest_path and self._manifest_path.exists():
            try:
                raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                self._manifest = {k: v for k, v in raw.items() if not k.startswith("_")}
            except (OSError, json.JSONDecodeError) as error:
                logger.warning(f"Could not read corpus manifest {self._manifest_path}: {error}")

    # -- live state ---------------------------------------------------------------
    def ingested_ids(self, force: bool = False) -> list[str]:
        with self._lock:
            if not force and self._cached_at and time.monotonic() - self._cached_at < self._ttl:
                return list(self._cached_ids)
            try:
                self._cached_ids = self._store.list_paper_ids()
                self._error = None
            except Exception as error:  # noqa: BLE001 — Qdrant down: keep last known list
                self._error = str(error)
                logger.warning(f"CorpusInfo could not list papers ({error}); using last known list")
            self._cached_at = time.monotonic()
            return list(self._cached_ids)

    @property
    def error(self) -> str | None:
        return self._error

    def is_available(self) -> bool:
        """False when the vector store is unreachable AND we have no cached list —
        the API reports that as 503 rather than answering from nothing."""
        return bool(self.ingested_ids()) or self._error is None

    def title_for(self, paper_id: str) -> str | None:
        entry = self._manifest.get(paper_id) or self._manifest.get(paper_id.split("v")[0]) or {}
        return entry.get("title")

    def categories_for(self, paper_id: str) -> list[str]:
        entry = self._manifest.get(paper_id) or {}
        return list(entry.get("categories") or [])

    def paper(self, paper_id: str) -> dict | None:
        ingested = set(self.ingested_ids())
        if paper_id not in ingested and paper_id not in self._manifest:
            return None
        entry = self._manifest.get(paper_id) or {}
        return {"paper_id": paper_id, "title": entry.get("title"),
                "categories": list(entry.get("categories") or []),
                "authors": list(entry.get("authors") or []),
                "published": entry.get("published"), "n_chunks": entry.get("n_chunks"),
                "ingested": paper_id in ingested}

    def papers(self, limit: int | None = None, offset: int = 0) -> list[dict]:
        ids = sorted(set(self.ingested_ids()))
        window = ids[offset: offset + limit] if limit is not None else ids[offset:]
        return [self.paper(pid) for pid in window]

    def count(self) -> int:
        return len(self.ingested_ids())

    def category_counts(self) -> dict[str, int]:
        counts: Counter = Counter()
        for pid in self.ingested_ids():
            for category in self.categories_for(pid):
                counts[category] += 1
        return dict(counts.most_common())

    # -- prompt-ready text ----------------------------------------------------------
    def describe(self, max_categories: int = 12, max_examples: int = 3) -> str:
        """Compact description injected into classifier / chit-chat / redirect prompts.
        Summarised, never paper-by-paper: the corpus can hold hundreds of papers."""
        ids = self.ingested_ids()
        if not ids:
            return "No papers are currently indexed."

        counts = self.category_counts()
        lines = [f"{len(ids)} arXiv paper(s) currently searchable."]
        if counts:
            fields = Counter()
            for category, n in counts.items():
                fields[field_of(category)] += n
            lines.append("Fields covered: " + ", ".join(f"{name} ({n})" for name, n in fields.most_common(8)) + ".")
            lines.append("Top arXiv categories: "
                         + ", ".join(f"{c} ({n})" for c, n in list(counts.items())[:max_categories]) + ".")

        examples = [t for t in (self.title_for(pid) for pid in ids[:max_examples * 4]) if t][:max_examples]
        if examples:
            lines.append("Example titles: " + "; ".join(f'"{t}"' for t in examples) + ".")
        return "\n".join(lines)

    def short_scope(self) -> str:
        n = self.count()
        if not n:
            return "the indexed arXiv papers (none are loaded yet)"
        counts = self.category_counts()
        if counts:
            fields = Counter()
            for category, c in counts.items():
                fields[field_of(category)] += c
            top = ", ".join(name for name, _ in fields.most_common(4))
            return f"{n} arXiv research papers (mainly {top})"
        return f"{n} indexed arXiv research papers"
