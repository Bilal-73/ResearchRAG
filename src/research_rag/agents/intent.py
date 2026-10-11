"""Intent classification: the routing decision at the top of the graph.

  chit_chat | follow_up | new_question    (the three real routes)
  out_of_scope                            (a fourth: without it, "write me a
                                           web scraper" would be mis-filed as
                                           new_question and sent through
                                           retrieval pointlessly)

Layers, cheapest first:
  1. rule    — bare greetings/thanks skip the LLM entirely (free, instant)
  2. llm     — JSON classification with corpus + recent-turn context
  3. guards  — no history => follow_up is impossible; low confidence or an
               unknown label => new_question, the conservative path (retrieval
               + grounded refusal can't do harm, whereas mis-routing a real
               question to chit-chat would)
  4. heuristic fallback if the LLM call fails or returns junk
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from research_rag.llm import LLMClient
from .prompts import CLASSIFIER_SYSTEM

logger = logging.getLogger(__name__)

INTENTS = ("follow_up", "new_question", "chit_chat", "out_of_scope")

_GREETING = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|yo|good (morning|afternoon|evening)|thanks?( you)?( so much)?|thank you( so much)?|"
    r"thx|ty|ok(ay)?|cool|great|nice|bye|goodbye|see you|salam|assalam[\w\- ]*)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
# Requests no research paper can answer — personal advice or "do my work for me".
_OUT_OF_SCOPE = re.compile(
    r"\b(write|build|debug|fix|refactor|generate)\b[^.?]{0,30}\b(me|my|a|an)\b[^.?]{0,30}"
    r"\b(script|program|code|function|app|website|essay|poem|email|blog post)\b|"
    r"\bshould i\b[^.?]{0,40}\b(take|buy|sell|invest|sue|quit|see a doctor)\b|"
    r"\bwhat'?s the weather\b|\bwho won\b[^.?]{0,30}\b(match|election|game)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_CUE = re.compile(
    r"^(and|but|so|also|what about|how about|why|how so|tell me more|more (details|detail|info)|explain|"
    r"can you (explain|elaborate|clarify|simplify)|what does (that|this|it) mean|in (that|this|the same)|"
    r"which (one|method|model|dataset)|was (it|that|this)|is (it|that|this)|did (it|they)|were (they|those|these))\b",
    re.IGNORECASE,
)
_PRONOUN = re.compile(r"\b(it|its|that|this|they|them|those|these|the same|former|latter)\b", re.IGNORECASE)


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str
    source: str                 # "rule" | "llm" | "heuristic"
    injection_flag: bool = False


def is_greeting(text: str) -> bool:
    return bool(_GREETING.match(text))


class IntentClassifier:
    def __init__(self, llm: LLMClient, corpus_info=None, min_confidence: float = 0.5):
        self._llm = llm
        self._corpus = corpus_info
        self._min_confidence = min_confidence

    def classify(self, message: str, *, recent_turns: str = "", summary: str = "") -> IntentResult:
        has_history = bool(recent_turns or summary)

        if is_greeting(message):
            return IntentResult("chit_chat", 0.99, "bare greeting/thanks", "rule")

        corpus_text = self._corpus.describe() if self._corpus else "(unknown)"
        context = []
        if summary:
            context.append(f"SUMMARY OF EARLIER CONVERSATION:\n{summary}")
        if recent_turns:
            context.append(f"RECENT TURNS:\n{recent_turns}")
        context_block = "\n\n".join(context) or "(none — this is the first message)"

        raw = self._llm.chat_json(
            [
                {"role": "system", "content": CLASSIFIER_SYSTEM.format(corpus=corpus_text)},
                {"role": "user", "content": f"CONVERSATION CONTEXT:\n{context_block}\n\nLATEST USER MESSAGE:\n{message}"},
            ],
            purpose="classify", default={}, max_tokens=120,
        )

        intent = str(raw.get("intent", "")).strip().lower()
        if intent not in INTENTS:
            logger.warning(f"Classifier returned unusable output {raw!r} — using heuristic fallback")
            return self._heuristic(message, has_history)

        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.7))))
        except (TypeError, ValueError):
            confidence = 0.7
        reason = str(raw.get("reason", ""))[:120]
        injection = bool(raw.get("is_injection_attempt", False))

        if intent == "follow_up" and not has_history:
            intent, reason = "new_question", "no earlier conversation to follow up on"
        if confidence < self._min_confidence and intent in ("chit_chat", "out_of_scope", "follow_up"):
            reason = f"low confidence ({confidence:.2f}) in {intent}; defaulting to a grounded search"
            intent = "new_question"

        return IntentResult(intent, confidence, reason, "llm", injection)

    def _heuristic(self, message: str, has_history: bool) -> IntentResult:
        if _OUT_OF_SCOPE.search(message):
            return IntentResult("out_of_scope", 0.6, "non-research request phrasing", "heuristic")
        words = message.split()
        if has_history and (_FOLLOW_UP_CUE.match(message.strip()) or (len(words) <= 8 and _PRONOUN.search(message))):
            return IntentResult("follow_up", 0.55, "short/anaphoric message with prior context", "heuristic")
        return IntentResult("new_question", 0.5, "default", "heuristic")
