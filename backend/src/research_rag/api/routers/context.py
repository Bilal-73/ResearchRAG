"""Context-engineering introspection: what the system caches, and what it remembers.

These two endpoints expose the halves of context engineering that are otherwise
invisible from outside the process:

* `GET /v1/cache` — the Redis layer. Embeddings are deterministic for a given
  text, so a warm cache turns a repeated question (or a re-run of ingestion)
  into zero embedding API calls. Without counters you cannot tell whether that
  is actually happening, so hits/misses/hit-rate and per-namespace key counts
  are reported, along with Redis's own memory and uptime.
* `GET /v1/chat/{conversation_id}/memory` — the agent's working context. Only
  the last N turns are kept verbatim; older ones are folded into a rolling LLM
  summary so the prompt stays flat however long the conversation runs. This
  shows exactly what the next turn's prompt will be built from.

Neither endpoint returns cached *values* (embedding vectors are large and the
cached text is the user's), only counts and the conversation's own content.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from research_rag.api.dependencies import AppState, get_state, require_api_key
from research_rag.api.schemas import (
    CacheNamespace, CacheResponse, CacheStats, ErrorResponse, MemoryResponse, MemoryTurn,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["context"])


@router.get("/cache", response_model=CacheResponse,
            responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
            summary="Redis cache state and hit/miss counters")
async def cache(
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
) -> CacheResponse:
    """Counters are per-process and reset when the API restarts; key counts come
    from Redis and are shared across processes."""
    if state.cache is None:
        return CacheResponse(enabled=False, url="", ttl_seconds=0, key_prefix="",
                             stats=CacheStats(), namespaces=[], server={},
                             note="No cache configured (REDIS_CACHE_ENABLED=false).")

    info = state.cache.info()
    note = None
    if not info["enabled"]:
        note = (f"Redis at {info['url']} is unreachable — the system is running without a cache, "
                f"so every embedding is recomputed. Start one with: "
                f"docker run -d -p 6379:6379 --name redis-server redis")
    return CacheResponse(
        enabled=info["enabled"], url=info["url"], ttl_seconds=info["ttl_seconds"],
        key_prefix=info["key_prefix"], stats=CacheStats(**info["stats"]),
        namespaces=[CacheNamespace(**ns) for ns in info["namespaces"]],
        server=info["server"], note=note,
    )


@router.post("/cache/reset-stats", status_code=status.HTTP_204_NO_CONTENT,
             responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
             summary="Zero the hit/miss counters (keeps cached data)")
async def reset_cache_stats(
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
):
    """Useful for a clean cold-vs-warm demo: reset, ask a question, ask it again,
    and compare. Cached entries themselves are untouched."""
    if state.cache is not None:
        state.cache.reset_stats()
    return None


@router.get("/chat/{conversation_id}/memory", response_model=MemoryResponse,
            responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
                       503: {"model": ErrorResponse}},
            summary="What the agent remembers about a conversation")
async def memory(
    conversation_id: str,
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
) -> MemoryResponse:
    snapshot = state.agent.memory_snapshot(conversation_id)
    history = snapshot.get("history") or []
    if not history and not snapshot.get("summary"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "conversation_not_found",
                    "message": f"No stored memory for conversation '{conversation_id}'. "
                               f"It may have expired, been deleted, or never existed.",
                    "detail": {"conversation_id": conversation_id}},
        )
    focus = snapshot.get("focus") or {}
    return MemoryResponse(
        conversation_id=conversation_id,
        turns=snapshot.get("turns", 0),
        summary=snapshot.get("summary", ""),
        focus_question=focus.get("question", ""),
        focus_papers=list(focus.get("papers") or []),
        history=[MemoryTurn(role=m.get("role", ""), content=m.get("content", ""),
                            intent=m.get("intent", ""))
                 for m in history],
        max_turns_kept=state.settings.memory_max_turns,
    )
