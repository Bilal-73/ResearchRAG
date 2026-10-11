"""Output-side guardrails: pure functions over the generated answer.

The LLM-based groundedness check lives in the agent (it needs the LLM client);
everything deterministic lives here so it is trivial to unit-test.

The reference check is the scientific-corpus analogue of a hallucination
guard: models are fluent at inventing plausible arXiv ids and DOIs, and a
fabricated identifier is far more damaging than vague prose because it looks
verifiable. Any identifier in the answer that is not present in the retrieved
excerpts or their paper ids is stripped and reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .injection import Canary
from .pii import OUTPUT_PII_KINDS, PIIGuard
from .input_guard import event

STANDARD_REFUSAL = "This is not reported in the retrieved context."

# Any of these means the model declined instead of asserting facts, so a
# missing citation is expected and must not be flagged.
_REFUSAL_MARKERS = (
    "not reported in the retrieved context", "not reported in the", "does not report",
    "not mentioned", "not stated", "not specified", "not provided", "not covered",
    "no information", "cannot find", "unable to find", "does not contain", "not available in",
)

# Warnings the output guard appends to an answer for the reader's benefit. They are
# annotations ABOUT the answer, not part of it, and every one begins a paragraph
# with this marker.
ANNOTATION_MARKER = "⚠️"
_ANNOTATION_BLOCK = re.compile(
    r"\n{2,}" + re.escape(ANNOTATION_MARKER) + r".*?(?=\n{2,}|$)", re.DOTALL)


def strip_annotations(answer: str) -> str:
    """The model's answer, without the guardrail warnings appended to it.

    Needed when an answer is handed to an automatic judge. RAGAS's
    answer_relevancy treats a hedged or disclaimed answer as "noncommittal" and
    scores it 0, so leaving our own "this may not be fully supported" warning in
    place penalises the system precisely for being honest about its uncertainty —
    measured at 0.55 -> 0.00 on one answer. The warning still reaches the user;
    it just does not reach the judge.
    """
    return _ANNOTATION_BLOCK.sub("", answer).strip()


LEAK_REPLACEMENT = ("I can't share that. I can answer questions about the findings in the indexed "
                    "research papers — what would you like to know?")

_BRACKET_GROUP = re.compile(r"\[([^\[\]]{1,40})\]")
_LABEL = re.compile(r"S(\d+)", re.IGNORECASE)

# Identifiers a model might fabricate. arXiv: 2410.14077 / 2410.14077v2 (and the
# pre-2007 "math.GT/0309136" form); DOI: 10.xxxx/...
#
# The month part is constrained to 01-12 on purpose: a loose \d{4}\.\d{4,5} also
# matches ordinary decimals like "1234.5678", and stripping a legitimate number out
# of an answer is a worse failure than missing one fabricated id.
_ARXIV_REF = re.compile(r"\b(?:arxiv:\s*)?(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
_ARXIV_OLD_REF = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b")
_DOI_REF = re.compile(r"\b(10\.\d{4,9}/[^\s,;)\]]+)", re.IGNORECASE)


@dataclass
class CitationCheck:
    text: str                                   # answer with invalid markers removed
    used: list[str] = field(default_factory=list)      # ["S1", "S3"] in order of first use
    invalid: list[str] = field(default_factory=list)
    is_refusal: bool = False

    @property
    def ok(self) -> bool:
        """An answer is fine if it cites something valid, or if it's an explicit refusal."""
        return bool(self.used) or self.is_refusal


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    """Find [S#] markers (also [S1, S2]), drop any that point at a source that
    doesn't exist, and report which valid ones were used."""
    used: list[str] = []
    invalid: list[str] = []

    def _replace(match: re.Match) -> str:
        inner = match.group(1)
        labels = _LABEL.findall(inner)
        # Only treat it as a citation group if it is made of S-labels and separators
        if not labels or _LABEL.sub("", inner).strip(" ,;/&") != "":
            return match.group(0)
        kept = []
        for number in labels:
            label = f"S{int(number)}"
            if 1 <= int(number) <= n_sources:
                kept.append(label)
                if label not in used:
                    used.append(label)
            else:
                invalid.append(label)
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _BRACKET_GROUP.sub(_replace, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).replace(" .", ".").replace(" ,", ",")
    return CitationCheck(text=cleaned, used=used, invalid=invalid, is_refusal=is_refusal(cleaned))


def find_references(text: str) -> set[str]:
    """Every arXiv id / DOI mentioned in a piece of text, normalised to lower case."""
    found = {m.group(1) for m in _ARXIV_REF.finditer(text)}
    found |= {m.group(1) for m in _ARXIV_OLD_REF.finditer(text)}
    found |= {m.group(1).rstrip(".") for m in _DOI_REF.finditer(text)}
    return {ref.lower() for ref in found}


def _grounded_reference_set(paper_ids: list[str], excerpts: list[str]) -> set[str]:
    """Identifiers the answer is allowed to mention: the retrieved papers' own ids
    (with and without the version suffix) plus anything the excerpts themselves cite."""
    allowed: set[str] = set()
    for pid in paper_ids:
        allowed.add(pid.lower())
        allowed.add(re.sub(r"v\d+$", "", pid).lower())
    for text in excerpts:
        allowed |= find_references(text)
        allowed |= {re.sub(r"v\d+$", "", ref) for ref in find_references(text)}
    return allowed


def check_references(answer: str, paper_ids: list[str], excerpts: list[str]) -> tuple[str, list[str]]:
    """Strip arXiv ids / DOIs that appear nowhere in the retrieved evidence.

    Returns the cleaned answer and the list of fabricated identifiers removed."""
    allowed = _grounded_reference_set(paper_ids, excerpts)
    fabricated = sorted(
        ref for ref in find_references(answer)
        if ref not in allowed and re.sub(r"v\d+$", "", ref) not in allowed
    )
    if not fabricated:
        return answer, []

    cleaned = answer
    for ref in fabricated:
        # Remove the identifier and any "arXiv:" prefix / empty brackets it leaves behind.
        cleaned = re.sub(rf"\(?\s*(?:arxiv:\s*)?{re.escape(ref)}\s*\)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).replace(" .", ".").replace(" ,", ",")
    return cleaned.strip(), fabricated


class OutputGuardrails:
    def __init__(self, pii: PIIGuard, canary: Canary, reference_check_enabled: bool = True):
        self.pii = pii
        self.canary = canary
        self.reference_check_enabled = reference_check_enabled

    def check_leak(self, answer: str) -> tuple[str, list[dict]]:
        if self.canary.leaked_in(answer):
            return LEAK_REPLACEMENT, [event("prompt_leak", "blocked", "canary token found in the answer — answer replaced")]
        return answer, []

    def scan_pii(self, answer: str) -> tuple[str, list[dict]]:
        result = self.pii.redact(answer, kinds=OUTPUT_PII_KINDS)
        if result.found:
            return result.text, [event("output_pii", "redacted", f"masked {result.summary()} in the answer")]
        return answer, []

    def check_references(self, answer: str, paper_ids: list[str], excerpts: list[str]) -> tuple[str, list[dict]]:
        if not self.reference_check_enabled:
            return answer, []
        cleaned, fabricated = check_references(answer, paper_ids, excerpts)
        if not fabricated:
            return answer, []
        return cleaned, [event("reference_check", "stripped",
                               f"identifier(s) not present in the retrieved evidence: {', '.join(fabricated)}")]
