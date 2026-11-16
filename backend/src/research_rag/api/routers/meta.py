"""Health and guardrail introspection.

`/health` is deliberately dependency-free (no auth, no agent requirement) so an
orchestrator can always reach it, and it reports *per component* rather than a
single boolean — "degraded" with `vector_store: down` is actionable in a way
that a bare 503 is not. It returns 200 when serving and 503 only when the API
genuinely cannot answer questions, which is what a readiness probe should key on.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response, status

from research_rag.api.dependencies import API_VERSION, AppState, get_state, require_api_key
from research_rag.api.schemas import ComponentHealth, GuardrailInfo, GuardrailsResponse, HealthResponse
from research_rag.guardrails import STAGE_TITLES, STAGES

logger = logging.getLogger(__name__)

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependency status")
async def health(request: Request, response: Response) -> HealthResponse:
    state: AppState = request.app.state.services
    components: dict[str, ComponentHealth] = {}

    if not state.ready:
        components["agent"] = ComponentHealth(status="down", detail=state.build_error or "not built")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="down", version=API_VERSION, components=components)

    components["agent"] = ComponentHealth(status="ok")

    n_papers = 0
    try:
        n_papers = state.corpus.count()
        if state.corpus.error:
            components["vector_store"] = ComponentHealth(status="down", detail=state.corpus.error)
        elif n_papers == 0:
            components["vector_store"] = ComponentHealth(
                status="degraded", detail="reachable but no papers indexed — run scripts/run_ingestion.py")
        else:
            components["vector_store"] = ComponentHealth(status="ok", detail=f"{n_papers} paper(s) indexed")
    except Exception as error:  # noqa: BLE001 — health must never raise
        components["vector_store"] = ComponentHealth(status="down", detail=str(error))

    stats = getattr(state.llm, "stats", {}) or {}
    llm_status = "degraded" if stats.get("failures") else "ok"
    components["llm"] = ComponentHealth(
        status=llm_status,
        detail=f"calls={stats.get('calls', 0)} retries={stats.get('retries', 0)} "
               f"rate_limit_hits={stats.get('rate_limit_hits', 0)} failures={stats.get('failures', 0)}",
    )
    if state.cache is None:
        components["cache"] = ComponentHealth(status="degraded", detail="no cache configured")
    elif state.cache.enabled:
        cache_stats = state.cache.stats()
        components["cache"] = ComponentHealth(
            status="ok",
            detail=f"redis; hits={cache_stats['hits']} misses={cache_stats['misses']}"
                   + (f" hit_rate={cache_stats['hit_rate']:.0%}" if cache_stats["hit_rate"] is not None else ""),
        )
    else:
        components["cache"] = ComponentHealth(
            status="degraded", detail="Redis unreachable — embeddings are recomputed every time")

    components["auth"] = ComponentHealth(
        status="ok" if state.settings.api_key_list else "degraded",
        detail="API key required" if state.settings.api_key_list else "OPEN — no API_KEYS configured",
    )

    worst = "ok"
    for component in components.values():
        if component.status == "down":
            worst = "down"
            break
        if component.status == "degraded":
            worst = "degraded"

    # "degraded" still serves traffic; only a hard "down" fails the readiness probe.
    if worst == "down":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status=worst, version=API_VERSION, components=components, n_papers=n_papers)


@router.get("/v1/guardrails", response_model=GuardrailsResponse, summary="Guardrails and their live state")
async def guardrails(
    state: AppState = Depends(get_state),
    api_key: str = Depends(require_api_key),
) -> GuardrailsResponse:
    """Rendered from the single catalogue in guardrails/registry.py, with each
    entry's `enabled` flag read from live settings — so this can't claim a
    guardrail is on when its setting is off."""
    return GuardrailsResponse(
        stages=STAGES, stage_titles=STAGE_TITLES,
        guardrails=[GuardrailInfo(**item) for item in state.registry.describe()],
    )
