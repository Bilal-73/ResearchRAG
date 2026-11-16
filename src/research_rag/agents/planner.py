"""Planner agent: splits a multi-part question into self-contained sub-questions.

Why: one embedding of "what dose was used AND what adverse effects occurred?" lands between two topics and tends
to retrieve neither well. Retrieving per part, then answering all parts together, fixes that.

Cost control: a cheap rule decides whether the question *might* be compound; only then is the LLM asked.
Ordinary single questions therefore pay nothing. Fail-safe: any failure -> no plan -> normal single-query path.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from research_rag.llm import LLMClient
from .prompts import PLANNER_PROMPT

logger = logging.getLogger(__name__)

_WH = re.compile(r"\b(what|which|who|when|where|why|how|did|does|do|was|were|is|are)\b", re.IGNORECASE)
_JOINER = re.compile(r"(\band\b|\balso\b|\bas well as\b|[,;])", re.IGNORECASE)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference(s)? between|differ(s|ed)? from|respectively)\b", re.IGNORECASE)
MAX_SUB_QUESTION_CHARS = 300


@dataclass
class PlanResult:
    sub_questions: list[str] = field(default_factory=list)
    reason: str = ""
    used_llm: bool = False

    @property
    def is_compound(self) -> bool:
        return len(self.sub_questions) > 1


def looks_compound(question: str) -> bool:
    if question.count("?") >= 2 or _COMPARE.search(question):
        return True
    return len(_WH.findall(question)) >= 2 and bool(_JOINER.search(question))


class QuestionPlanner:
    def __init__(self, llm: LLMClient, max_sub_questions: int = 3):
        self._chain = llm.json_chain(PLANNER_PROMPT, purpose="plan", default={}, max_tokens=250)
        self.max_sub_questions = max(1, max_sub_questions)

    def plan(self, question: str) -> PlanResult:
        if not looks_compound(question):
            return PlanResult(reason="single question (rule check)")

        raw = self._chain.invoke({"max_n": self.max_sub_questions, "question": question})
        subs, seen = [], set()
        for item in raw.get("sub_questions") or []:
            if not isinstance(item, str):
                continue
            text = " ".join(item.split())[:MAX_SUB_QUESTION_CHARS]
            if text and text.lower() not in seen:
                seen.add(text.lower())
                subs.append(text)
        subs = subs[: self.max_sub_questions]
        reason = str(raw.get("reason", ""))[:120]

        if len(subs) < 2:
            logger.info("Planner kept the question whole")
            return PlanResult(reason=reason or "planner kept it as one question", used_llm=True)
        logger.info(f"Planner split the question into {len(subs)} sub-questions")
        return PlanResult(subs, reason, used_llm=True)
