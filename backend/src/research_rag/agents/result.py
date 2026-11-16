"""What the agent hands back to callers (UI, CLI, tests)."""
from __future__ import annotations

from dataclasses import dataclass, field

from research_rag.models import Chunk, SourceCitation


@dataclass
class AgentResult:
    answer: str
    answer_kind: str                      # rag | chitchat | out_of_scope | not_found | blocked | safety | error
    intent: str = ""
    intent_confidence: float = 0.0
    intent_reason: str = ""
    intent_source: str = ""
    sources: list[SourceCitation] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    standalone_query: str = ""
    alternatives: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    plan_reason: str = ""
    plan_coverage: list[dict] = field(default_factory=list)
    hyde_passage: str = ""
    variants_used: list[str] = field(default_factory=list)
    n_candidates: int = 0
    retry_count: int = 0
    context_ok: bool = True
    context_reason: str = ""
    citations_ok: bool = True
    grounded: bool | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)      # guardrail events for this turn
    trace: list[dict] = field(default_factory=list)       # per-node timing
    blocked: bool = False
    block_kind: str = ""
    clean_message: str = ""               # the PII-redacted text that was actually processed
    thread_id: str = ""
    error: str = ""

    @property
    def total_seconds(self) -> float:
        return sum(step.get("seconds", 0.0) for step in self.trace)

    @property
    def cited_sources(self) -> list[SourceCitation]:
        return [s for s in self.sources if s.cited]
