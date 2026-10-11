"""One resilient wrapper around the Azure OpenAI chat endpoint.

Every agent step (intent classification, query rewriting, HyDE, grading,
generation, groundedness check ...) goes through this class, so upstream
rate-limit handling lives in exactly one place:

  * retries on 429 / 5xx / timeouts / connection errors with exponential
    backoff + jitter, honouring the server's ``Retry-After`` header;
  * a semaphore caps simultaneous in-flight calls (several Streamlit users
    share one Azure quota);
  * counters (calls, retries, rate-limit hits, failures) that the UI shows.

This is *upstream* rate-limit management. The per-user request limiter that
protects the app itself is in ``guardrails/rate_limit.py``.
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when a call still fails after every retry."""


class LLMClient:
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        deployment: str | None = None,
        *,
        client=None,                      # inject a fake in tests
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 20.0,
        max_concurrency: int = 4,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if client is None:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
                timeout=timeout, max_retries=0,   # we own the retry policy, not the SDK
            )
        self._client = client
        self.deployment = deployment
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "retries": 0, "rate_limit_hits": 0, "failures": 0}

    # ------------------------------------------------------------------ public
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
        purpose: str = "llm",
    ) -> str:
        kwargs: dict = {"model": self.deployment, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        attempt = 0
        while True:
            try:
                with self._semaphore:
                    t0 = time.monotonic()
                    response = self._client.chat.completions.create(**kwargs)
                    elapsed = time.monotonic() - t0
                self._bump("calls")
                text = response.choices[0].message.content or ""
                logger.info(f"LLM[{purpose}] ok in {elapsed:.1f}s (attempt {attempt + 1}, {len(text)} chars)")
                return text
            except Exception as error:  # noqa: BLE001 — classified below
                status = getattr(error, "status_code", None)
                retryable = status in _RETRYABLE_STATUS or _is_transient(error)
                if status == 429:
                    self._bump("rate_limit_hits")
                if not retryable or attempt >= self._max_retries:
                    self._bump("failures")
                    logger.error(f"LLM[{purpose}] failed permanently after {attempt + 1} attempt(s): {error}")
                    raise LLMError(f"LLM call '{purpose}' failed: {error}") from error

                delay = self._backoff_delay(attempt, error)
                self._bump("retries")
                logger.warning(f"LLM[{purpose}] transient error (status={status}): {error}. "
                               f"Retry {attempt + 1}/{self._max_retries} in {delay:.1f}s")
                self._sleep(delay)
                attempt += 1

    def chat_json(
        self,
        messages: list[dict],
        *,
        purpose: str,
        default: dict | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Ask for a JSON object. Never raises on bad JSON — returns ``default``
        (or {}) so a flaky classifier can't take the whole graph down; the
        caller decides what a safe fallback looks like."""
        try:
            raw = self.chat(messages, json_mode=True, purpose=purpose, max_tokens=max_tokens)
        except LLMError:
            return dict(default or {})
        parsed = parse_json_object(raw)
        if parsed is None:
            logger.warning(f"LLM[{purpose}] returned unparseable JSON: {raw[:120]!r}")
            return dict(default or {})
        return parsed

    # ---------------------------------------------------------------- internals
    def _bump(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _backoff_delay(self, attempt: int, error: Exception) -> float:
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return min(retry_after, self._max_delay)
        exponential = min(self._base_delay * (2 ** attempt), self._max_delay)
        return exponential * (0.5 + random.random() / 2)  # jitter: 50-100% of the step


def _is_transient(error: Exception) -> bool:
    name = type(error).__name__
    return name in {"APIConnectionError", "APITimeoutError", "Timeout", "ConnectError", "ReadTimeout"}


def _retry_after_seconds(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(raw: str) -> dict | None:
    """Tolerant JSON-object parser: strips code fences and falls back to the
    first {...} span if the model wrapped the JSON in prose."""
    text = _FENCE_RE.sub("", raw.strip())
    for candidate in (text, _first_brace_span(text)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _first_brace_span(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else None
