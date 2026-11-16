# guardrails/

- **`registry.py`** — catalogue of every guardrail; `GET /v1/guardrails` renders from it, and each
  entry's `enabled` flag is read from live settings, so the API cannot claim a guardrail is on when
  its setting is off.
- **`pii.py`** — regex PII masking + optional Presidio NER for names. Tuned to leave scientific
  notation alone: `2410.14077v2`, `10.1145/3292500`, `p=0.03`, `95% CI 0.8-1.4` must survive, or
  the assistant mangles the questions it exists to answer.
- **`injection.py`** — Unicode-normalised injection scoring, retrieved-chunk scan, output canary.
  Note the deliberate omission of `context` from the instruction-override rule: "models that ignore
  prior context" is ordinary ML-paper prose, and this rule also scans retrieved chunks, so
  including it silently dropped legitimate evidence.
- **`safety.py`** — `SelfHarmDetector` (crisis language → fixed support message) and
  `DualUseDetector` (procedural weapons/mass-harm requests → declined). The dual-use patterns
  require an *action* aimed at a weapons-class category, never a topic word alone: "what detection
  method does this paper propose?" must pass, since discussing hazardous literature is the whole
  point of the tool.
- **`rate_limit.py`** — sliding-window limiter, per API client plus global, in-memory; swap the
  storage for Redis if you run replicas. Keyed to the caller, not the conversation, so starting a
  new conversation can't reset a quota.
- **`output_guard.py`** — citation validation, PII scan, prompt-leak canary, and the
  **fabricated-reference check**: arXiv ids and DOIs in an answer that appear nowhere in the
  retrieved evidence are stripped and reported. An invented-but-plausible identifier is the most
  damaging hallucination on a scientific corpus precisely because it looks verifiable.
- **`input_guard.py`** — orchestrator; order is rate → length → PII → self-harm → dual-use →
  injection, cheapest and most safety-critical first.

Everything here is deterministic and unit-tested. The LLM-side rules in
`generation/prompt_builder.py` and the groundedness check in `agents/nodes.py` are the other half.
