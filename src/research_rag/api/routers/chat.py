"""POST /v1/chat — the one endpoint a frontend really needs.

Status codes are chosen so a frontend can branch on the code alone:

  200  the agent produced a turn. This includes refusals, out-of-scope
       redirects and safety responses: the request was valid and the system
       behaved correctly, so `answer_kind` carries the nuance, not the status.
  400  the request is malformed in a way validation can't express (empty
       message after trimming).
  401  API_KEYS is configured and the key is missing or wrong.
  413  the message exceeds MAX_INPUT_CHARS — the payload itself is the problem.
  422  FastAPI's own schema validation (wrong types, missing fields).
  429  rate limited, with Retry-After in both the header and the body.
  503  the agent could not be built, or the vector store / LLM is unreachable.
  500  an unexpected failure; the body carries a stable code, never a traceback.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from starlette.concurrency import run_in_threadpool

from research_rag.agents import agents_called, route_decisions, route_path
from research_rag.guardrails import strip_annotations
from research_rag.api.dependencies import AppState, client_id, get_state, require_api_key
from research_rag.api.schemas import (
    AnswerScores, ChatRequest, ChatResponse, Diagnostics, ErrorResponse, GuardrailEvent, Source, TraceStep,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])

RESPONSES = {
    400: {"model": ErrorResponse, "description": "Empty or unusable message"},
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    413: {"model": ErrorResponse, "description": "Message longer than the configured input cap"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded; see Retry-After"},
    503: {"model": ErrorResponse, "description": "Assistant or a dependency is unavailable"},
}


def _to_response(result, include_diagnostics: bool, scores: AnswerScores | None = None) -> ChatResponse:
    sources = [
        Source(
            label=s.label or f"S{i}", paper_id=s.paper_id, paper_title=s.paper_title,
            section=s.section, section_id=s.section_id, page_no=s.page_no,
            chunk_type=s.chunk_type, figure_no=s.figure_no, snippet=s.snippet,
            score=s.score, cited=s.cited, url=s.url,
        )
        for i, s in enumerate(result.sources, start=1)
    ]
    diagnostics = None
    if include_diagnostics:
        rows = {row["node"]: row for row in agents_called(result)}
        diagnostics = Diagnostics(
            intent=result.intent, intent_confidence=result.intent_confidence,
            intent_reason=result.intent_reason, intent_source=result.intent_source,
            standalone_query=result.standalone_query, sub_questions=result.sub_questions,
            variants_used=result.variants_used, n_candidates=result.n_candidates,
            retry_count=result.retry_count, context_ok=result.context_ok,
            context_reason=result.context_reason, citations_ok=result.citations_ok,
            grounded=result.grounded, unsupported_claims=result.unsupported_claims,
            route=route_path(result), explanation=route_decisions(result),
            trace=[
                TraceStep(node=step["node"],
                          role=rows.get(step["node"], {}).get("role", ""),
                          kind=rows.get(step["node"], {}).get("kind", ""),
                          detail=step.get("detail", ""), seconds=round(step.get("seconds", 0.0), 3))
                for step in result.trace
            ],
            total_seconds=round(result.total_seconds, 3),
        )
    return ChatResponse(
        answer=result.answer, answer_kind=result.answer_kind or "rag",
        conversation_id=result.thread_id, sources=sources,
        blocked=result.blocked, block_kind=result.block_kind or "",
        guardrail_events=[GuardrailEvent(**e) for e in result.events],
        diagnostics=diagnostics,
        scores=scores,
    )


@router.post("/chat", response_model=ChatResponse, responses=RESPONSES,
             summary="Ask a question about the indexed papers")
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    include_diagnostics: bool = True,
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
) -> ChatResponse:
    """Runs one turn of the guardrailed agent.

    Pass `conversation_id` from a previous response to continue a conversation;
    omit it to start a fresh one. Set `include_diagnostics=false` to drop the
    trace from the payload once a frontend no longer needs it.

    Set `score: true` in the body to have RAGAS score the answer that comes back
    (faithfulness and answer relevancy) and return it in `scores`. It is off by
    default because it roughly doubles the turn: measured at ~15-20s of extra
    latency on gpt-4o-mini, on top of the answer itself.
    """
    message = payload.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "empty_message", "message": "Send a non-empty question."},
        )

    settings = state.settings
    if len(message) > settings.max_input_chars:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"error": "message_too_long",
                    "message": f"Message is {len(message)} characters; the limit is {settings.max_input_chars}.",
                    "detail": {"length": len(message), "limit": settings.max_input_chars}},
        )

    conversation_id = payload.conversation_id or state.agent.new_thread_id()
    options = payload.options.as_dict() if payload.options else {}
    result = state.agent.chat(message, conversation_id, options,
                              client_id=client_id(request, api_key))

    if result.block_kind == "rate_limit":
        retry_after = _retry_after(result)
        response.headers["Retry-After"] = str(int(retry_after) or 1)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "rate_limited", "message": result.answer,
                    "retry_after_seconds": retry_after},
            headers={"Retry-After": str(int(retry_after) or 1)},
        )
    if result.block_kind == "too_long":
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"error": "message_too_long", "message": result.answer},
        )
    if result.answer_kind == "error":
        logger.error(f"Agent error: {result.error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "upstream_unavailable", "message": result.answer},
        )

    scores = await _score_answer(payload, state, message, result)
    return _to_response(result, include_diagnostics, scores)


async def _score_answer(payload, state, question: str, result) -> AnswerScores | None:
    """Reference-free RAGAS scoring of this turn, when the caller opts in.

    Skipped unless the turn actually produced a grounded answer: scoring a
    chit-chat reply or a refusal against retrieved context is meaningless, and
    there is no context to score it against anyway.

    Runs in a worker thread because RAGAS is synchronous and would otherwise
    block the event loop for every other request for several seconds.
    """
    if not payload.score:
        return None
    if not state.settings.enable_inference_scoring:
        return AnswerScores(error="Inference scoring is disabled on this server "
                                  "(ENABLE_INFERENCE_SCORING=false).")
    if result.answer_kind != "rag" or not result.chunks:
        return AnswerScores(error=f"Not scored: this turn produced '{result.answer_kind}', "
                                  f"which has no retrieved context to score against.")

    scorer = _get_scorer(state)
    if scorer is None:
        return AnswerScores(error="RAGAS is not installed on the server. "
                                  "Install it with: pip install -r requirements-eval.txt")

    t0 = time.monotonic()
    # Judge the model's answer, not the guardrail warnings appended to it — see
    # guardrails.strip_annotations for why that distinction changes the score.
    scored = await run_in_threadpool(
        scorer.score, question, strip_annotations(result.answer), [c.text for c in result.chunks])
    return AnswerScores(
        faithfulness=scored.get("faithfulness"),
        answer_relevancy=scored.get("answer_relevancy"),
        seconds=round(time.monotonic() - t0, 2),
        error=scored.get("error"),
    )


def _get_scorer(state):
    """Lazily build the RAGAS scorer once, on the first request that asks for it —
    so a server nobody scores against never pays to construct the judge models,
    and a server that does pay only once."""
    if "inference_scorer" not in state.extras:
        try:
            from research_rag.evaluation.ragas_eval import InferenceScorer

            state.extras["inference_scorer"] = InferenceScorer(state.settings)
            logger.info("Inference-time RAGAS scorer ready")
        except ImportError as error:
            logger.warning(f"RAGAS unavailable, inference scoring disabled: {error}")
            state.extras["inference_scorer"] = None
    return state.extras["inference_scorer"]


def _retry_after(result) -> float:
    for event in result.events:
        if event.get("guardrail") == "rate_limit":
            detail = event.get("detail", "")
            if "retry in ~" in detail:
                try:
                    return float(detail.split("retry in ~")[1].rstrip("s"))
                except (IndexError, ValueError):
                    break
    return 60.0


@router.delete("/chat/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, responses=RESPONSES,
               summary="Forget a conversation")
async def delete_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
) -> Response:
    """Drops the stored memory for a conversation. Idempotent: deleting an unknown
    conversation is still 204, since the end state the caller wanted is the one they get."""
    state.agent.reset_thread(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
