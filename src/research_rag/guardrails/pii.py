"""PII detection + redaction.

Runs on the user's message BEFORE it reaches any LLM, the conversation memory
or the logs (data minimisation: the raw value is never stored anywhere; findings
keep only the *type*). Also runs on the final answer.

Default backend is deterministic regex — zero extra dependencies, fast, and
easy to reason about. It is tuned to leave scientific notation alone:
"2410.14077v2", "10.1145/3292500", "p=0.03", "95% CI 0.8-1.4", "n=62" and
version/equation numbers must survive untouched, or the assistant would
mangle the very questions it exists to answer.

Known limit of the regex backend: it only catches a person's *name* when
they introduce themselves ("my name is ..."). Free-floating names need NER;
set PII_BACKEND=presidio (see requirements-extras.txt) for that. Trade-off:
NER also masks author names inside research questions ("what did Vaswani et al.
report?"), which can weaken author-based lookups — hence regex is the default.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    kind: str


@dataclass
class PIIResult:
    text: str                                   # redacted text
    counts: dict[str, int] = field(default_factory=dict)   # {"EMAIL": 1, ...} — types only, never values

    @property
    def found(self) -> bool:
        return bool(self.counts)

    def summary(self) -> str:
        return ", ".join(f"{n}× {kind}" for kind, n in sorted(self.counts.items()))


# ---- patterns ---------------------------------------------------------------
# (kind, compiled regex, group to redact — 0 = whole match)
_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0),
    # Explicit international format: +92 300 1234567, +1 (415) 555-2671
    ("PHONE", re.compile(r"(?<![\w+])\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?(?:[\s.\-]?\d{2,4}){2,4}(?!\d)"), 0),
    # US style with separators: 415-555-2671, (415) 555 2671
    ("PHONE", re.compile(r"(?<![\d\-])\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?![\d\-])"), 0),
    # Pakistani mobile: 0300-1234567, 03001234567
    ("PHONE", re.compile(r"(?<![\d\-])03\d{2}[\s\-]?\d{7}(?!\d)"), 0),
    ("SSN", re.compile(r"(?<![\d\-])\d{3}-\d{2}-\d{4}(?![\d\-])"), 0),
    ("NATIONAL_ID", re.compile(r"(?<![\d\-])\d{5}-\d{7}-\d(?![\d\-])"), 0),   # Pakistan CNIC
    ("IP_ADDRESS", re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"), 0),
    (
        "DATE_OF_BIRTH",
        re.compile(
            r"(?i:\b(?:dob|d\.o\.b\.?|date of birth|birth ?date|born on)\b)\s*[:\-]?\s*"
            r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}|\d{1,2} [A-Za-z]{3,9},? \d{4})"
        ),
        1,
    ),
    (
        "PERSON_NAME",
        re.compile(r"(?i:\bmy name is)\s+([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,2})"),
        1,
    ),
]

_CARD_CANDIDATE = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _regex_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for kind, pattern, group in _PATTERNS:
        for match in pattern.finditer(text):
            spans.append(_Span(match.start(group), match.end(group), kind))
    for match in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            spans.append(_Span(match.start(), match.end(), "CREDIT_CARD"))
    return spans


def _resolve_overlaps(spans: list[_Span]) -> list[_Span]:
    """Longest span wins; ties go to the earlier span."""
    chosen: list[_Span] = []
    for span in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if all(span.end <= c.start or span.start >= c.end for c in chosen):
            chosen.append(span)
    return sorted(chosen, key=lambda s: s.start)


class PIIGuard:
    def __init__(self, backend: str = "regex", enabled: bool = True):
        self.enabled = enabled
        self.backend = backend
        self._presidio = None
        if enabled and backend == "presidio":
            self._presidio = _load_presidio()
            if self._presidio is None:
                logger.warning("PII_BACKEND=presidio requested but Presidio/spaCy is not installed — "
                               "falling back to the regex backend. See requirements-extras.txt.")
                self.backend = "regex"

    def redact(self, text: str, kinds: set[str] | None = None) -> PIIResult:
        """Mask PII. ``kinds`` restricts which types are masked (used for the
        output scan, where context-based types like DATE_OF_BIRTH make no sense)."""
        if not self.enabled or not text:
            return PIIResult(text=text)

        spans = _regex_spans(text)
        if self._presidio is not None:
            spans.extend(self._presidio(text))
        if kinds is not None:
            spans = [s for s in spans if s.kind in kinds]
        spans = _resolve_overlaps(spans)
        if not spans:
            return PIIResult(text=text)

        pieces, cursor = [], 0
        for span in spans:
            pieces.append(text[cursor:span.start])
            pieces.append(f"[{span.kind}]")
            cursor = span.end
        pieces.append(text[cursor:])
        counts = dict(Counter(s.kind for s in spans))
        return PIIResult(text="".join(pieces), counts=counts)


# Types worth scanning for in an *answer* (the corpus is public research, so
# context-dependent input types are not applied there).
OUTPUT_PII_KINDS = {"EMAIL", "PHONE", "SSN", "NATIONAL_ID", "CREDIT_CARD", "IP_ADDRESS"}


# ---- optional Presidio adapter ------------------------------------------------
def _load_presidio():
    """Returns a callable text -> list[_Span], or None if Presidio isn't installed.

    Only person names are taken from Presidio (regex already covers the
    structured types with medical-safe tuning). A high score threshold keeps
    method names such as 'Kalman' from being mistaken for people.

    Uses the SMALL spaCy model (en_core_web_sm, ~12 MB). Presidio's own default
    is the 400 MB en_core_web_lg — set PRESIDIO_SPACY_MODEL=en_core_web_lg to
    use it for better recall."""
    import os

    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)   # it warns about every unused language recogniser
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except Exception:  # noqa: BLE001
        return None
    model = os.environ.get("PRESIDIO_SPACY_MODEL", "en_core_web_sm")
    try:
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}],
        }).create_engine()
        engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    except Exception as error:  # noqa: BLE001 — usually the spaCy model is missing
        logger.warning(f"Presidio installed but could not start ({error}). "
                       f"Run: pip install {model} (see requirements-extras.txt)")
        return None

    def analyze(text: str) -> list[_Span]:
        results = engine.analyze(text=text, language="en", entities=["PERSON"], score_threshold=0.75)
        return [_Span(r.start, r.end, "PERSON_NAME") for r in results]

    return analyze
