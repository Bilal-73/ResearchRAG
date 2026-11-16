"""Runs every input-side guardrail in a fixed, deliberate order.

  1. rate limit      — cheapest check, and it must come first so a flood of
                       requests can't cost us CPU or tokens
  2. length cap      — before any regex/LLM touches a huge string
  3. PII redaction   — everything after this point only ever sees redacted text
  4. self-harm       — safety beats everything else that follows
  5. dual-use        — procedural weapons requests, declined before retrieval
  6. prompt injection

Returns one InputVerdict; the LangGraph `guard_input` node just copies it into
state. Events carry only *types and counts* — never the raw PII or full text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .injection import PromptInjectionGuard
from .pii import PIIGuard
from .rate_limit import SlidingWindowRateLimiter
from .safety import DualUseDetector, SelfHarmDetector

logger = logging.getLogger(__name__)


def event(guardrail: str, action: str, detail: str = "") -> dict:
    """Uniform shape for everything shown in the API's guardrail report."""
    return {"guardrail": guardrail, "action": action, "detail": detail}


@dataclass
class InputVerdict:
    clean_text: str
    blocked: bool = False
    block_kind: str | None = None        # rate_limit | too_long | empty | self_harm | dual_use | injection
    message: str = ""                    # what to tell the user when blocked
    suspicious: bool = False             # injection flagged but not blocked -> harden downstream
    events: list[dict] = field(default_factory=list)
    store_in_memory: bool = True         # blocked-for-abuse turns are not remembered
    retry_after_seconds: float | None = None   # set on rate_limit, surfaced as the HTTP Retry-After header


class InputGuardrails:
    def __init__(
        self,
        rate_limiter: SlidingWindowRateLimiter,
        pii: PIIGuard,
        injection: PromptInjectionGuard,
        self_harm: SelfHarmDetector,
        dual_use: DualUseDetector,
        max_input_chars: int = 1500,
    ):
        self.rate_limiter = rate_limiter
        self.pii = pii
        self.injection = injection
        self.self_harm = self_harm
        self.dual_use = dual_use
        self.max_input_chars = max_input_chars

    def check(self, text: str, session_id: str) -> InputVerdict:
        text = text or ""
        events: list[dict] = []

        decision = self.rate_limiter.check_and_consume(session_id)
        if not decision.allowed:
            events.append(event("rate_limit", "blocked", f"{decision.limit_name}; retry in ~{decision.retry_after_seconds:.0f}s"))
            return InputVerdict("", True, "rate_limit", decision.message(), events=events, store_in_memory=False,
                                retry_after_seconds=decision.retry_after_seconds)

        if not text.strip():
            return InputVerdict("", True, "empty", "Please send a question about the indexed research papers.",
                                events=events, store_in_memory=False)

        if len(text) > self.max_input_chars:
            events.append(event("input_length", "blocked", f"{len(text)} chars > limit {self.max_input_chars}"))
            return InputVerdict(
                "", True, "too_long",
                f"That message is too long ({len(text)} characters). Please shorten it to "
                f"{self.max_input_chars} characters or fewer.",
                events=events, store_in_memory=False,
            )

        pii = self.pii.redact(text)
        clean = pii.text
        if pii.found:
            events.append(event("pii_redaction", "redacted", f"masked {pii.summary()} before processing"))

        crisis = self.self_harm.check(clean)
        if crisis.triggered:
            events.append(event("self_harm", "blocked", "crisis language detected — support message returned"))
            return InputVerdict(clean, True, "self_harm", crisis.message, events=events)

        hazard = self.dual_use.check(clean)
        if hazard.triggered:
            events.append(event("dual_use", "blocked", "procedural weapons/mass-harm request declined"))
            logger.warning("Dual-use request declined before retrieval")
            return InputVerdict(clean, True, "dual_use", hazard.message, events=events, store_in_memory=False)

        scan = self.injection.scan(clean)
        if scan.level == "high":
            events.append(event("prompt_injection", "blocked", f"score {scan.score}; families: {', '.join(scan.families)}"))
            logger.warning(f"Prompt injection blocked (score={scan.score}, families={scan.families})")
            return InputVerdict(
                clean, True, "injection",
                "I can't act on that message because it looks like an attempt to change my instructions. "
                "I'm here to answer questions about the indexed research papers — feel free to ask one.",
                events=events, store_in_memory=False,
            )
        suspicious = scan.level == "suspicious"
        if suspicious:
            events.append(event("prompt_injection", "flagged", f"score {scan.score}; families: {', '.join(scan.families)} — prompt hardened"))

        return InputVerdict(clean, suspicious=suspicious, events=events)
