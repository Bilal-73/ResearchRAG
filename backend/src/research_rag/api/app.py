"""FastAPI application factory.

Three decisions worth stating:

* **Startup never hard-fails.** The agent is built in the lifespan handler; if
  it throws, the error is stored and the app still serves `/health` and
  `/docs`, with every other endpoint returning 503 and the reason. A container
  that won't boot gives an operator nothing to read.
* **Every error body has one shape** (`ErrorResponse`): `{error, message,
  detail?, retry_after_seconds?}`. Handlers below convert FastAPI's validation
  errors and any uncaught exception into it, so a frontend writes one error
  path instead of three. Tracebacks are logged, never returned.
* **Every response carries `X-Request-ID`**, echoed from the request when the
  caller supplies one, so a user-visible error can be matched to a log line.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from research_rag.api.dependencies import API_VERSION, AppState
from research_rag.api.routers import chat, context, corpus, meta

logger = logging.getLogger(__name__)

DESCRIPTION = """
Agentic, guardrailed RAG over an **arXiv research-paper corpus**
([Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench), arXiv split).

Text, tables and figures are all retrievable, and every answer is returned with
system-generated citations down to the paper **section** — the same granularity the
benchmark judges, so retrieval quality is measurable rather than asserted.

**Start here:** `POST /v1/chat`. Pass the `conversation_id` you get back to continue a
conversation. `GET /v1/corpus` says what is indexed; `GET /v1/guardrails` lists every
guardrail and whether it is live; `GET /health` reports per-component status.

**Blocked and refused turns return 200**, not an error: the request was valid and the
system behaved correctly. Branch on `answer_kind` (`rag`, `chitchat`, `out_of_scope`,
`not_found`, `blocked`, `safety`) and the `blocked` flag. Genuine errors use 4xx/5xx and
always carry `{error, message}`.
"""

TAGS_METADATA = [
    {"name": "chat", "description": "Ask questions and manage conversations."},
    {"name": "corpus", "description": "Inspect which papers are indexed."},
    {"name": "context", "description": "Context engineering: Redis cache state and conversation memory."},
    {"name": "meta", "description": "Health checks and guardrail introspection."},
]


def _error(status_code: int, code: str, message: str, detail=None, retry_after=None) -> JSONResponse:
    body = {"error": code, "message": message}
    if detail is not None:
        body["detail"] = detail
    if retry_after is not None:
        body["retry_after_seconds"] = retry_after
    headers = {"Retry-After": str(int(retry_after) or 1)} if retry_after else None
    return JSONResponse(status_code=status_code, content=body, headers=headers)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from config.logging_config import setup_logging
    from config.settings import get_settings

    settings = get_settings()
    setup_logging(settings.log_level)
    state = AppState(settings=settings)
    app.state.services = state

    try:
        from research_rag.bootstrap import build_agent

        agent, registry, rate_limiter, llm, corpus_info, cache = build_agent()
        state.agent, state.registry = agent, registry
        state.rate_limiter, state.llm, state.corpus, state.cache = rate_limiter, llm, corpus_info, cache
        logger.info(f"API ready — {corpus_info.count()} paper(s) indexed")
    except Exception as error:  # noqa: BLE001 — serve /health instead of refusing to boot
        state.build_error = f"{type(error).__name__}: {error}"
        logger.exception("Agent construction failed; the API will serve /health and return 503 elsewhere")

    yield
    logger.info("API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research RAG API",
        description=DESCRIPTION,
        version=API_VERSION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    from config.settings import get_settings

    origins = get_settings().cors_origin_list or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,   # credentials + wildcard origin is rejected by browsers
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms", "Retry-After"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(f"[{request_id}] unhandled error on {request.method} {request.url.path}")
            response = _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error",
                              "Something went wrong handling this request.")
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{(time.monotonic() - start) * 1000:.0f}"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Routers raise HTTPException with a dict detail in ErrorResponse shape;
        anything else (404s from the router table, etc.) is normalised into it."""
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            body = dict(exc.detail)
        else:
            body = {"error": _code_for(exc.status_code), "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=body, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "validation_error",
                      "The request body did not match the expected schema.",
                      detail={"errors": exc.errors()[:10]})

    app.include_router(meta.router)
    app.include_router(chat.router)
    app.include_router(corpus.router)
    app.include_router(context.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"name": "Research RAG API", "version": API_VERSION, "docs": "/docs", "health": "/health"}

    return app


def _code_for(status_code: int) -> str:
    return {
        400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
        405: "method_not_allowed", 413: "message_too_long", 422: "validation_error",
        429: "rate_limited", 500: "internal_error", 503: "service_unavailable",
    }.get(status_code, "error")


app = create_app()
