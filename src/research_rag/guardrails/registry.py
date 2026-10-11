"""Single catalogue of every guardrail in the system.

`GET /v1/guardrails` renders straight from this list, so what the API claims
is implemented and what the code actually does cannot drift apart: a guardrail
is only reported as active if the setting that switches it on is on, and the
per-turn "events" in a chat response come from real guardrail events emitted
at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass

# Pipeline stages, in the order they run.
STAGES = ["input", "routing", "retrieval", "generation", "output", "system"]
STAGE_TITLES = {
    "input": "Input",
    "routing": "Routing",
    "retrieval": "Retrieval",
    "generation": "Generation",
    "output": "Output",
    "system": "System & resilience",
}


@dataclass(frozen=True)
class GuardrailSpec:
    id: str
    name: str
    stage: str
    description: str
    implementation: str      # where to look in the code
    setting: str | None = None   # Settings attribute that toggles it (None = always on)


GUARDRAIL_SPECS: list[GuardrailSpec] = [
    # ---- input ----------------------------------------------------------------
    GuardrailSpec(
        "rate_limit", "Rate limiting", "input",
        "Sliding-window limits per API client (per minute and per hour) plus a global "
        "per-minute cap. A blocked request returns HTTP 429 with a Retry-After header.",
        "guardrails/rate_limit.py",
    ),
    GuardrailSpec(
        "input_length", "Input length cap", "input",
        "Rejects oversized messages before any model sees them (cost control and a "
        "common injection-smuggling vector). Returns HTTP 413.",
        "guardrails/input_guard.py",
    ),
    GuardrailSpec(
        "pii_redaction", "PII redaction", "input",
        "Emails, phone numbers, national IDs, card numbers, IPs, dates of birth and "
        "self-stated names are masked BEFORE the text reaches any LLM, memory or log. "
        "Scientific notation like 2410.14077v2, 10.1145/3292500 or p=0.03 is left alone.",
        "guardrails/pii.py", "pii_redaction_enabled",
    ),
    GuardrailSpec(
        "self_harm", "Crisis-language detection", "input",
        "First-person self-harm language short-circuits the whole pipeline with a calm "
        "message pointing to local crisis support. No retrieval, no LLM.",
        "guardrails/safety.py", "self_harm_guard_enabled",
    ),
    GuardrailSpec(
        "dual_use", "Dual-use request screening", "input",
        "The corpus spans q-bio, chemistry and physics. Discussing hazardous research is in "
        "scope; procedural requests to produce weapons or mass-harm agents are declined "
        "before retrieval. Topic words alone never trigger it — a procedural ask is required.",
        "guardrails/safety.py", "hazard_guard_enabled",
    ),
    GuardrailSpec(
        "prompt_injection", "Prompt-injection detection", "input",
        "Unicode-normalised pattern scoring for instruction override, role hijack, "
        "system-prompt extraction, fake delimiters and obfuscation. High scores are "
        "blocked; borderline ones are flagged and the prompt is hardened.",
        "guardrails/injection.py", "injection_guard_enabled",
    ),
    # ---- routing --------------------------------------------------------------
    GuardrailSpec(
        "intent_routing", "Intent classification & redirection", "routing",
        "An LLM classifier labels each message follow-up / new question / chit-chat / "
        "out of scope, and the graph routes accordingly. Chit-chat and off-topic "
        "messages never trigger retrieval.",
        "agents/intent.py",
    ),
    GuardrailSpec(
        "scope_redirect", "Out-of-scope redirect", "routing",
        "Requests the corpus cannot serve — writing code to order, general trivia, "
        "opinion pieces — get a deterministic redirect naming what the corpus does cover.",
        "agents/nodes.py",
    ),
    # ---- retrieval ------------------------------------------------------------
    GuardrailSpec(
        "retrieved_content_scan", "Retrieved-content injection scan", "retrieval",
        "Indirect injection defence: every retrieved chunk is scanned, and chunks that "
        "look like instructions are dropped before they reach the generator. Relevant for "
        "a corpus of arbitrary uploaded papers.",
        "guardrails/injection.py", "injection_guard_enabled",
    ),
    GuardrailSpec(
        "context_grading", "Context sufficiency grading", "retrieval",
        "An agent step checks whether the retrieved evidence can actually answer the "
        "question and triggers one corrective retrieval pass if it cannot.",
        "agents/nodes.py", "enable_context_grading",
    ),
    # ---- generation -----------------------------------------------------------
    GuardrailSpec(
        "grounded_prompt", "Grounded, citation-only prompt", "generation",
        "The model may only use the numbered context excerpts, must cite each claim as "
        "[S#], must say 'not reported' instead of guessing, and treats excerpts as data, "
        "never instructions.",
        "generation/prompt_builder.py",
    ),
    # ---- output ---------------------------------------------------------------
    GuardrailSpec(
        "citation_enforcement", "Citation enforcement", "output",
        "Answers must carry valid [S#] markers. Invalid markers are stripped, an uncited "
        "answer is regenerated once, and if it is still uncited it is visibly flagged.",
        "guardrails/output_guard.py",
    ),
    GuardrailSpec(
        "reference_check", "Fabricated-reference check", "output",
        "arXiv ids and DOIs are the easiest thing for a model to invent convincingly. Any "
        "identifier in the answer that does not appear in the retrieved evidence is stripped "
        "and reported.",
        "guardrails/output_guard.py", "reference_check_enabled",
    ),
    GuardrailSpec(
        "groundedness", "Groundedness check", "output",
        "A second LLM pass verifies every claim against the retrieved excerpts and "
        "flags anything unsupported.",
        "agents/nodes.py", "enable_groundedness_check",
    ),
    GuardrailSpec(
        "output_pii", "Output PII scan", "output",
        "The final answer is scanned for PII with the same detectors as the input.",
        "guardrails/pii.py", "pii_redaction_enabled",
    ),
    GuardrailSpec(
        "prompt_leak", "System-prompt leak canary", "output",
        "A random canary token lives in the system prompt; if it ever appears in an "
        "answer, the answer is replaced.",
        "guardrails/injection.py",
    ),
    # ---- system ---------------------------------------------------------------
    GuardrailSpec(
        "llm_backoff", "Upstream rate-limit handling", "system",
        "Azure 429 / 5xx / timeouts are retried with exponential backoff and jitter, "
        "honouring Retry-After. A semaphore caps concurrent LLM calls.",
        "llm/client.py",
    ),
    GuardrailSpec(
        "bounded_loops", "Bounded agent loops", "system",
        "The corrective-retrieval loop is capped (MAX_RETRIEVAL_RETRIES), so the graph "
        "can never spin and burn quota.",
        "agents/graph.py",
    ),
    GuardrailSpec(
        "safe_degradation", "Fail-safe defaults", "system",
        "If the classifier, grader or checker fails, the system falls back to the "
        "conservative path (grounded retrieval / visible warning) instead of erroring "
        "or silently skipping a check. Upstream outages surface as HTTP 503, never as a "
        "confident answer.",
        "agents/nodes.py",
    ),
]


class GuardrailRegistry:
    """Combines the static catalogue with the live on/off state from Settings."""

    def __init__(self, settings=None):
        self._settings = settings

    def is_enabled(self, spec: GuardrailSpec) -> bool:
        if spec.setting is None or self._settings is None:
            return True
        return bool(getattr(self._settings, spec.setting, True))

    def describe(self) -> list[dict]:
        return [
            {
                "id": spec.id, "name": spec.name, "stage": spec.stage,
                "description": spec.description, "implementation": spec.implementation,
                "enabled": self.is_enabled(spec),
            }
            for spec in GUARDRAIL_SPECS
        ]

    def by_stage(self) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {stage: [] for stage in STAGES}
        for item in self.describe():
            grouped[item["stage"]].append(item)
        return grouped
