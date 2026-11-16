"""Shared API wiring: the lazily-built agent singleton, auth, and client identity.

The agent is expensive to construct (the cross-encoder load dominates), so it
is built once at startup and reused. If construction fails — Azure
misconfigured, Qdrant unreachable — the app still starts and serves
`/health` and the docs, and request endpoints return 503. A container that
refuses to boot tells an operator far less than one that boots and reports
exactly which component is down.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from typing import Any

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

API_VERSION = "2.0.0"


@dataclass
class AppState:
    """Everything built at startup, hung off `app.state.services`.

    Deliberately untyped (`Any`) rather than importing the concrete classes:
    this module is imported by every router, and pulling the agent, retriever
    and cross-encoder in at import time would make the API's startup cost the
    same whether or not it ever serves a request.
    """
    agent: Any = None
    registry: Any = None
    rate_limiter: Any = None
    llm: Any = None
    corpus: Any = None
    cache: Any = None
    settings: Any = None
    build_error: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.agent is not None


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.services
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "service_unavailable",
                    "message": "The assistant is not available. Check /health for which component is down.",
                    "detail": {"build_error": state.build_error}},
        )
    return state


def get_settings_dep(request: Request):
    return request.app.state.services.settings


async def require_api_key(request: Request, x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Optional bearer-style auth. With API_KEYS unset the API is open, which is
    fine for local development and wrong for anything else — /health says which
    mode is live so it can't be deployed open by accident unnoticed."""
    settings = request.app.state.services.settings
    keys = settings.api_key_list if settings else []
    if not keys:
        return "anonymous"

    provided = x_api_key or ""
    authorization = request.headers.get("authorization", "")
    if not provided and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    # compare_digest against every key: constant time, no early exit on prefix.
    if not any(secrets.compare_digest(provided, key) for key in keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "unauthorized", "message": "A valid X-API-Key header is required."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided


def client_id(request: Request, api_key: str) -> str:
    """Identity for rate limiting: the API key when authenticated, else the peer IP.

    X-Forwarded-For is honoured only when TRUST_FORWARDED_FOR is on — otherwise
    any client could spoof the header and reset its own rate-limit bucket.
    """
    if api_key and api_key != "anonymous":
        return f"key:{api_key[:12]}"
    settings = request.app.state.services.settings
    if settings and settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"
