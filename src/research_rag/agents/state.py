"""LangGraph state.

Two kinds of keys live here, and the split matters:

PERSISTED (survive between turns of one conversation via the checkpointer,
keyed by thread_id) — this is the agent's *memory*:
    history   recent turns, verbatim (redacted text only — never raw PII)
    summary   rolling summary of older turns that were folded out of `history`
    focus     what the last answer was about, for resolving "what about that?"

PER-TURN (reset to defaults at the start of every turn by `fresh_turn()`):
    everything else — routing decisions, retrieval artefacts, the draft
    answer, guardrail events, the trace.

Everything is plain JSON-serialisable data (dicts/lists/str) on purpose: it
keeps checkpoints portable across the in-memory and SQLite backends and
avoids registering custom classes with the serializer.
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ---- inputs ------------------------------------------------------------------
    user_message: str
    session_id: str         # rate-limit identity: the API client, NOT the conversation
    options: dict

    # ---- persisted memory ----------------------------------------------------------
    history: list[dict]      # [{"role": "user"|"assistant", "content": str, "intent": str}]
    summary: str
    focus: dict              # {"question": str, "papers": [paper_id, ...]}

    # ---- per-turn: guardrails --------------------------------------------------------
    clean_message: str
    blocked: bool
    block_kind: str
    suspicious: bool
    store_turn: bool

    # ---- per-turn: routing -----------------------------------------------------------
    intent: str
    intent_confidence: float
    intent_reason: str
    intent_source: str       # "llm" | "heuristic" | "rule"

    # ---- per-turn: retrieval ------------------------------------------------------------
    standalone_query: str
    sub_questions: list[str]      # planner output; empty = ordinary single-question path
    plan_reason: str
    plan_coverage: list[dict]     # [{"question": str, "sources": ["S1", ...]}]
    alternatives: list[str]
    hyde_passage: str
    retry_count: int
    next_action: str         # "retry" | "generate"
    n_candidates: int
    variants_used: list[str]
    top_chunks: list[dict]
    context_ok: bool
    context_reason: str

    # ---- per-turn: generation / output ---------------------------------------------------
    answer: str
    answer_kind: str         # rag | chitchat | out_of_scope | not_found | blocked | safety | error
    sources: list[dict]
    citations_ok: bool
    grounded: bool | None
    unsupported_claims: list[str]

    # ---- per-turn: observability ---------------------------------------------------------
    events: list[dict]
    trace: list[dict]
    error: str


def fresh_turn(user_message: str, session_id: str, options: dict) -> dict:
    """Input for one invocation. Providing every per-turn key here is what
    resets them — the persisted keys (history/summary/focus) are deliberately
    absent so the checkpointer's stored values carry over."""
    return {
        "user_message": user_message, "session_id": session_id, "options": options,
        "clean_message": "", "blocked": False, "block_kind": "", "suspicious": False, "store_turn": True,
        "intent": "", "intent_confidence": 0.0, "intent_reason": "", "intent_source": "",
        "standalone_query": "", "sub_questions": [], "plan_reason": "", "plan_coverage": [], "alternatives": [], "hyde_passage": "", "retry_count": 0,
        "next_action": "generate", "n_candidates": 0, "variants_used": [],
        "top_chunks": [], "context_ok": True, "context_reason": "",
        "answer": "", "answer_kind": "", "sources": [], "citations_ok": True,
        "grounded": None, "unsupported_claims": [],
        "events": [], "trace": [], "error": "",
    }
