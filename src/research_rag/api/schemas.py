"""Request/response models. These ARE the API contract a frontend codes against,
so every field is typed, documented and given an example.

Two conventions worth knowing:

* A blocked or refused turn is **not** an HTTP error. The request was valid and
  the system behaved correctly, so it returns 200 with `answer_kind` telling the
  frontend what happened (`safety`, `blocked`, `out_of_scope`, `not_found`) and
  a `blocked` flag. HTTP 4xx/5xx are reserved for things the caller or the
  server got wrong. The one exception is rate limiting, which is a true 429
  because clients and proxies are expected to act on it.
* Everything the agent did is exposed under `diagnostics`, so a UI can show the
  route, timings and guardrail events without a second endpoint.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AnswerKind = Literal["rag", "chitchat", "out_of_scope", "not_found", "blocked", "safety", "error"]
BlockKind = Literal["", "rate_limit", "too_long", "empty", "self_harm", "dual_use", "injection"]


# ---------------------------------------------------------------- chat ----------
class ChatOptions(BaseModel):
    """Per-request overrides of the retrieval pipeline. Omitted fields use server defaults,
    so a frontend can expose these as toggles for demos and A/B comparisons."""
    enable_query_rewrite: bool | None = None
    enable_hyde: bool | None = None
    enable_reranker: bool | None = None
    enable_planner: bool | None = None
    enable_context_grading: bool | None = None
    enable_groundedness_check: bool | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class AnswerScores(BaseModel):
    """RAGAS scores for THIS answer, when a request opts in with `score=true`.

    Only reference-free metrics appear here: the other RAGAS metrics
    (context_precision / context_recall) need a gold answer, which does not
    exist for a real user's question. A null score means the metric could not
    be computed — not that it scored zero."""
    faithfulness: float | None = Field(
        None, description="Is every claim in the answer supported by the retrieved context? 0-1.")
    answer_relevancy: float | None = Field(
        None, description="Does the answer address the question actually asked? 0-1.")
    seconds: float = Field(0.0, description="Added latency from scoring.")
    error: str | None = Field(None, description="Set when scoring could not run.")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000,
                         description="The user's message. Longer than the configured input cap returns 413.",
                         examples=["What problem does the VDF-RLS algorithm solve?"])
    conversation_id: str | None = Field(
        None, description="Returned by a previous call. Omit to start a new conversation; "
                          "memory, follow-up resolution and summarisation are scoped to it.",
        examples=["thread-9f2c1a7b4e01"])
    options: ChatOptions | None = None
    score: bool = Field(
        False,
        description="Score this answer with RAGAS (faithfulness, answer relevancy) and return it in "
                    "`scores`. Off by default: it adds ~15-20s of latency and several extra LLM calls.")


class Source(BaseModel):
    label: str = Field(..., description="Inline marker used in the answer, e.g. 'S1'.", examples=["S1"])
    paper_id: str = Field(..., examples=["2410.14077v2"])
    paper_title: str | None = None
    section: str = Field(..., examples=["IV. SIMULATION RESULTS"])
    section_id: int | None = Field(None, description="Section index within the paper, as the corpus defines it.")
    page_no: int | None = None
    chunk_type: Literal["text", "table", "figure"] = "text"
    figure_no: str | None = None
    snippet: str = Field(..., description="First ~200 characters of the excerpt.")
    score: float | None = Field(None, description="Reranker relevance score, when reranking ran.")
    cited: bool = Field(..., description="True if the answer references this source inline.")
    url: str | None = Field(None, examples=["https://arxiv.org/abs/2410.14077"])


class GuardrailEvent(BaseModel):
    guardrail: str = Field(..., examples=["pii_redaction"])
    action: str = Field(..., examples=["redacted"])
    detail: str = ""


class TraceStep(BaseModel):
    node: str = Field(..., examples=["retrieve"])
    role: str = Field("", description="Human-readable agent role for this step.", examples=["Retriever + Reranker"])
    kind: str = Field("", description="llm | rule | search | static | mixed")
    detail: str = ""
    seconds: float = 0.0


class Diagnostics(BaseModel):
    """Everything the agent did this turn — enough to render a full trace UI."""
    intent: str = ""
    intent_confidence: float = 0.0
    intent_reason: str = ""
    intent_source: str = Field("", description="rule | llm | heuristic")
    standalone_query: str = Field("", description="The rewritten, self-contained search query.")
    sub_questions: list[str] = []
    variants_used: list[str] = []
    n_candidates: int = 0
    retry_count: int = 0
    context_ok: bool = True
    context_reason: str = ""
    citations_ok: bool = True
    grounded: bool | None = Field(None, description="null = the check did not run or was unavailable.")
    unsupported_claims: list[str] = []
    route: list[str] = Field([], description="Graph nodes visited, in order.")
    explanation: list[str] = Field([], description="Plain-language reasons for the routing decisions.")
    trace: list[TraceStep] = []
    total_seconds: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    answer_kind: AnswerKind = Field(..., description="How the answer was produced; `rag` is the only grounded path.")
    conversation_id: str = Field(..., description="Pass this back to continue the conversation.")
    sources: list[Source] = []
    blocked: bool = False
    block_kind: BlockKind = ""
    guardrail_events: list[GuardrailEvent] = []
    diagnostics: Diagnostics | None = None
    scores: AnswerScores | None = Field(
        None, description="Present only when the request set `score=true` and the turn produced a "
                          "grounded answer with retrieved context.")


# ---------------------------------------------------------------- corpus --------
class PaperSummary(BaseModel):
    paper_id: str
    title: str | None = None
    categories: list[str] = []
    authors: list[str] = []
    published: str | None = None
    n_chunks: int | None = Field(None, description="Indexed chunks for this paper.")
    url: str | None = Field(None, description="arXiv abstract page.",
                            examples=["https://arxiv.org/abs/2410.14077v2"])
    pdf_url: str | None = Field(None, description="arXiv PDF. The corpus ships pre-parsed JSON, so "
                                                  "no PDF is stored locally — this links to arXiv.",
                                examples=["https://arxiv.org/pdf/2410.14077v2"])


class CorpusResponse(BaseModel):
    n_papers: int
    categories: dict[str, int] = Field({}, description="arXiv category -> number of indexed papers.")
    description: str = Field(..., description="The same corpus summary the agent's prompts receive.")
    papers: list[PaperSummary] = []
    total: int = Field(..., description="Total indexed papers, for pagination.")
    limit: int
    offset: int


# ---------------------------------------------------------------- context -------
class CacheStats(BaseModel):
    """Per-process counters since the API started (or the last stats reset)."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0
    lookups: int = 0
    hit_rate: float | None = Field(None, description="hits / (hits + misses); null before any lookup.")
    avg_get_ms: float | None = None
    avg_set_ms: float | None = None
    get_seconds: float = 0.0
    set_seconds: float = 0.0


class CacheNamespace(BaseModel):
    name: str = Field(..., examples=["embed"])
    label: str
    keys: int = Field(..., description="Entries currently cached in this namespace.")


class CacheResponse(BaseModel):
    enabled: bool = Field(..., description="False when Redis is unreachable or disabled by config.")
    url: str
    ttl_seconds: int
    key_prefix: str
    stats: CacheStats
    namespaces: list[CacheNamespace] = []
    server: dict = Field({}, description="Redis version, memory use, uptime, total keys.")
    note: str | None = Field(None, description="Set when caching is off, explaining why and how to fix it.")


class MemoryTurn(BaseModel):
    role: Literal["user", "assistant", ""] = ""
    content: str
    intent: str = ""


class MemoryResponse(BaseModel):
    """The agent's working context for one conversation."""
    conversation_id: str
    turns: int = Field(..., description="Exchanges kept verbatim.")
    max_turns_kept: int = Field(..., description="MEMORY_MAX_TURNS; older turns fold into the summary.")
    summary: str = Field("", description="Rolling LLM summary of turns that aged out of the history.")
    focus_question: str = ""
    focus_papers: list[str] = []
    history: list[MemoryTurn] = []


# ---------------------------------------------------------------- meta ----------
class GuardrailInfo(BaseModel):
    id: str
    name: str
    stage: str
    description: str
    implementation: str
    enabled: bool


class GuardrailsResponse(BaseModel):
    stages: list[str]
    stage_titles: dict[str, str]
    guardrails: list[GuardrailInfo]


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "down"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    components: dict[str, ComponentHealth]
    n_papers: int = 0


class ErrorResponse(BaseModel):
    """Every 4xx/5xx body has this shape, so a frontend can handle errors generically."""
    error: str = Field(..., description="Stable machine-readable code.", examples=["rate_limited"])
    message: str = Field(..., description="Human-readable explanation, safe to show a user.")
    detail: dict | None = None
    retry_after_seconds: float | None = None
