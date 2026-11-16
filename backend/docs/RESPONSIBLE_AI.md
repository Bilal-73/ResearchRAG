# Responsible AI notes

This assistant answers questions about published arXiv research papers. The corpus spans every
arXiv category — including fields where papers legitimately describe hazardous subject matter — and
it is served as a public HTTP API, so the design assumes some inputs (and some retrieved documents)
will be adversarial, and that some users will be in distress. This page states what the system does
about that, **and what it does not do**.

## What it is, and what it is not

- **Is:** a literature assistant. It reports what specific papers contain, with citations to the
  paper and section, and refuses when the evidence isn't there.
- **Is not:** a source of personal advice (medical, legal, financial), a code-writing assistant, or
  an authority on anything outside the indexed papers. Those are routed to a fixed out-of-scope
  reply without retrieval.
- **Is not** a judge of whether a paper is *correct*. It reports what a paper claims. A retrieved
  claim being well-cited here says nothing about whether the research replicates.

## Principles → implementation

| Principle | What the system does | Where |
|---|---|---|
| **Transparency** | Every claim carries an inline `[S#]` marker resolved to paper / section / relevance score, with an arXiv link. The API separates *cited* from *merely retrieved* sources and returns the rewritten query, the route taken, a plain-language explanation of each routing decision, and a per-step timing trace. | `generation/`, `api/schemas.py`, `agents/routing.py` |
| **Honesty about uncertainty** | Missing evidence produces "not reported in the retrieved context". A grader checks sufficiency before generating; a second pass checks each claim is supported. Failures become visible warnings, never silence. A checker that is *unavailable* reports "not checked" (`grounded: null`), not "passed". | `agents/nodes.py` |
| **No fabricated provenance** | arXiv ids and DOIs that appear nowhere in the retrieved evidence are stripped from the answer and reported as a guardrail event. An invented-but-plausible citation is the most damaging hallucination here precisely because it looks checkable. | `guardrails/output_guard.py` |
| **Accuracy / scope** | The model is told what is actually indexed (read live from Qdrant) and instructed to keep every result attached to the method, dataset and setting it was measured on, and never to merge numbers across papers or configurations. | `corpus/`, `generation/prompt_builder.py` |
| **Privacy / data minimisation** | PII is masked before any LLM call, before memory, and before logging. Only masked text is stored. Guardrail events record *types and counts*, never values. | `guardrails/pii.py` |
| **Safety — crisis** | First-person self-harm language short-circuits to a fixed support message. No retrieval, no LLM, so it cannot be argued out of it. The message avoids country-specific numbers because location is unknown, and avoids probing questions, per crisis-communication guidance. | `guardrails/safety.py` |
| **Safety — dual use** | Procedural requests to produce weapons or mass-harm agents are declined before retrieval, and the generation prompt forbids reformulating a paper's content into operational instructions regardless of what an excerpt contains. Discussion and summarisation of hazardous literature remain in scope — topic words alone never trigger the guard. | `guardrails/safety.py`, `generation/prompt_builder.py` |
| **Security** | Injection defence in three layers: input scan (Unicode-normalised), retrieved-chunk scan (indirect injection from paper text), and an output canary. Retrieved text is fenced and cannot spoof a citation label. | `guardrails/injection.py` |
| **Abuse / cost control** | Per-client and global rate limits (keyed to the API client, not the conversation, so a new conversation can't reset a quota), an input-length cap, bounded agent loops, upstream 429 backoff. | `guardrails/rate_limit.py`, `llm/client.py` |
| **Honest failure** | Upstream outages return HTTP 503 with an explicit message. There is no path where an unreachable index or LLM produces a confident answer. | `api/routers/chat.py` |
| **Fail-safe defaults** | Classifier failure → heuristic; low confidence → grounded search (not chit-chat); grader failure → generate under the strict refusal rule; retrieval outage → honest error. | `agents/` |
| **Verifiable claims about itself** | `GET /v1/guardrails` renders from the single catalogue in `guardrails/registry.py`, with each `enabled` flag read from live settings — the API cannot advertise a guardrail that is switched off. | `guardrails/registry.py` |

## Deployment responsibilities (the operator's, not the code's)

- **`API_KEYS` empty means no authentication.** That is appropriate locally and wrong anywhere
  else. `/health` reports which mode is live so an open deployment is visible, but nothing stops
  you shipping it open.
- **`TRUST_FORWARDED_FOR=true` outside a trusted proxy lets clients spoof their rate-limit
  identity.** It defaults to false.
- **`CORS_ORIGINS=*`** is a convenience default; set real origins before exposing a browser
  frontend.
- Rate limiting is in-process. Several replicas need a shared store — see `rate_limit.py`.

## Known limitations (please read)

1. **Partial corpus.** Only the papers you downloaded and indexed are searchable. The benchmark has
   1,000; a typical local run has far fewer. Absence of evidence *here* is not evidence of absence,
   and the assistant says so rather than guessing.
2. **The corpus is preprints.** arXiv is largely not peer-reviewed. The assistant reports what a
   paper claims, with no assessment of methodology, replication or retraction status.
3. **Parsing is inherited, not performed.** Text, tables and figures come pre-extracted from the
   benchmark. Any extraction error in the dataset — a garbled table, a missed figure — is inherited
   silently, and the system cannot know it happened.
4. **LLM-based checks are fallible.** The classifier, grader and groundedness checker are
   themselves LLM calls. They reduce risk; they do not eliminate it. Verify important facts in the
   cited paper, which is why every answer links to it.
5. **Heuristic guardrails.** Regex PII detection misses names that aren't self-introduced (use
   `PII_BACKEND=presidio` for NER, which in turn also masks author names in research questions).
   Injection and dual-use scoring are pattern-based, English-only, and evadable by a determined
   attacker; they are layers, not proofs.
6. **Self-preference bias in evaluation.** Both judges (RAGAS and the built-in one) run on the same
   model family as the generator, so neither is an independent opinion. The retrieval metrics are
   judge-free and should be trusted further.
7. **Automatic judges penalise honesty.** RAGAS answer_relevancy scores a hedged answer as
   "noncommittal" = 0, so the groundedness warning this system appends to an uncertain answer
   dropped its own score from 0.55 to 0.00 until those annotations were excluded from what the
   judge sees (`guardrails.strip_annotations`). Any metric applied to a system that admits
   uncertainty deserves the same scrutiny.
8. **English only.**

## How to check it still behaves

- `python -m pytest tests -q` — 200+ offline tests, including adversarial inputs, scientific
  strings that must *not* be masked or blocked, hazardous-topic questions that must *not* be
  refused, poisoned retrieved chunks, canary leaks, fabricated citations, outages of each LLM step,
  API status codes, and memory isolation between conversations.
- `python scripts/run_evaluation.py` — retrieval metrics, answer quality against the benchmark's
  own reference answers, and refusal accuracy on questions whose evidence was deliberately left
  out of the index.
