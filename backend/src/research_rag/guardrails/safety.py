"""Content-safety checks that short-circuit the graph before any LLM call.

Two detectors, both deterministic (instant, free, and they can't be argued out
of their decision):

SelfHarmDetector  Any public chatbot meets people in crisis. First-person
                  self-harm language gets a calm, fixed support message — no
                  retrieval, no LLM. The reply avoids country-specific numbers
                  (we don't know where the person is) and probing questions.

DualUseDetector   An arXiv corpus spans q-bio, chemistry and physics, so the
                  assistant will be asked about hazardous subject matter.
                  Discussing or summarising that literature is in scope —
                  "what detection method does this paper propose?" — so the
                  patterns deliberately require a *procedural* request ("how do
                  I make/build/synthesise ...") aimed at a weapons-class
                  category word. Topic words alone never trigger, which also
                  keeps unrelated ML jargon ("gradient explosion", "explosive
                  growth") from being caught.

Both are heuristics, and the generation prompt's scope rules are the other
half of the defence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---- self-harm -------------------------------------------------------------------
_SELF_HARM = [
    r"\bi (want|wanna|am going|'m going|plan|planning|intend|need) to (die|kill myself|end (it|my life)|hurt myself|harm myself)\b",
    r"\bi(?:'m| am) (thinking|thinking about|going to|about to) (kill(ing)? myself|end(ing)? (it|my life)|suicid\w*)\b",
    r"\b(suicidal|suicide plan|kill myself|end my life|take my own life)\b",
    r"\bi (don'?t|do not) want to (live|be alive|be here)( anymore| any more)?\b",
    r"\bi (overdosed|have overdosed|just overdosed)\b",
]
_SELF_HARM_RE = [re.compile(p, re.IGNORECASE) for p in _SELF_HARM]

SELF_HARM_MESSAGE = (
    "I'm really sorry you're going through something this painful. You don't have to handle it alone, "
    "and it matters that you get support from a person right now.\n\n"
    "**Please contact your local emergency number or a crisis line in your country now**, or reach out to "
    "someone you trust and ask them to stay with you. If you have taken something or hurt yourself, "
    "seek emergency care immediately.\n\n"
    "I'm a research-literature tool and can't give you the support you deserve here. "
    "I'm glad to help with questions about papers whenever you're ready."
)

# ---- dual-use requests ------------------------------------------------------------
# A procedural ask ...
_PROCEDURAL = (
    r"(how (?:do|can|would) (?:i|we|you)|how to|give me|provide|write|tell me|explain how to|"
    r"steps? (?:to|for)|instructions? (?:to|for)|recipe for|protocol for|guide (?:to|for))"
)
# ... aimed at a weapons-class category (categories only — never specific agents).
_WEAPON_CLASS = (
    r"(chemical weapons?|biological weapons?|bioweapons?|nerve agents?|nuclear weapons?|"
    r"radiological weapons?|dirty bombs?|explosive devices?|improvised explosives?|bombs?|"
    r"weaponi[sz]ed? (?:pathogens?|agents?|viruses|bacteria)|enhanced pathogens?)"
)
_ACTION = r"(make|build|synthesi[sz]e|produce|manufacture|construct|assemble|acquire|weaponi[sz]e|enhance|culture|grow)"

_DUAL_USE_RE = [
    re.compile(rf"{_PROCEDURAL}\b[^.?\n]{{0,60}}\b{_ACTION}\b[^.?\n]{{0,40}}\b{_WEAPON_CLASS}", re.IGNORECASE),
    re.compile(rf"\b{_ACTION}\s+(?:a|an|my own|your own)?\s*{_WEAPON_CLASS}", re.IGNORECASE),
    re.compile(rf"{_PROCEDURAL}\b[^.?\n]{{0,40}}\b{_WEAPON_CLASS}\b[^.?\n]{{0,30}}\b(at home|from scratch|yourself|myself)\b",
               re.IGNORECASE),
]

DUAL_USE_MESSAGE = (
    "I can't help with that. I answer questions about what the indexed research papers report, and that "
    "doesn't extend to instructions for producing weapons or other things designed to cause mass harm — "
    "regardless of which paper the request points at.\n\n"
    "If you're after the research itself (what a paper measured, which method it proposes, what it concluded), "
    "ask that and I'll answer it with citations."
)


@dataclass
class SafetyResult:
    kind: str | None            # "self_harm" | "dual_use" | None
    message: str = ""

    @property
    def triggered(self) -> bool:
        return self.kind is not None


class SelfHarmDetector:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, text: str) -> SafetyResult:
        if not self.enabled or not text:
            return SafetyResult(None)
        if any(p.search(text) for p in _SELF_HARM_RE):
            return SafetyResult("self_harm", SELF_HARM_MESSAGE)
        return SafetyResult(None)


class DualUseDetector:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, text: str) -> SafetyResult:
        if not self.enabled or not text:
            return SafetyResult(None)
        if any(p.search(text) for p in _DUAL_USE_RE):
            return SafetyResult("dual_use", DUAL_USE_MESSAGE)
        return SafetyResult(None)
