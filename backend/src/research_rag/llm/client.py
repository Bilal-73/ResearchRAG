"""One resilient wrapper around a LangChain chat model.

Every agent step (intent classification, query rewriting, HyDE, grading,
generation, groundedness check ...) is an LCEL chain built here —
``prompt | llm.runnable(...) | parser`` — so upstream rate-limit handling
lives in exactly one place:

  * retries on 429 / 5xx / timeouts / connection errors with exponential
    backoff + jitter, honouring the server's ``Retry-After`` header;
  * a semaphore caps simultaneous in-flight calls (several API users
    share one Azure quota);
  * counters (calls, retries, rate-limit hits, failures) that /health shows.

The model itself is any ``BaseChatModel`` (``AzureChatOpenAI`` in production,
a scripted fake in tests); it is built with ``max_retries=0`` because this
class owns the retry policy. Each call runs with ``run_name=<purpose>``, so a
LangSmith / callback trace names every step.

This is *upstream* rate-limit management. The per-user request limiter that
protects the app itself is in ``guardrails/rate_limit.py``.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .parsers import JsonObjectOutputParser, parse_json_object

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

MessagesLike = PromptValue | Sequence[BaseMessage] | Sequence[dict]


class LLMError(RuntimeError):
    """Raised when a call still fails after every retry."""


class LLMClient:
    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 20.0,
        max_concurrency: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.chat_model = chat_model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "retries": 0, "rate_limit_hits": 0, "failures": 0}

    @property
    def deployment(self) -> str | None:
        return getattr(self.chat_model, "deployment_name", None) or getattr(self.chat_model, "model_name", None)

    # ------------------------------------------------------------------ public
    def chat(
        self,
        messages: MessagesLike,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
        purpose: str = "llm",
    ) -> str:
        lc_messages = _to_messages(messages)
        kwargs: dict = {"temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        model = self.chat_model.bind(**kwargs).with_config(run_name=purpose, tags=[purpose])

        attempt = 0
        while True:
            try:
                with self._semaphore:
                    t0 = time.monotonic()
                    response = model.invoke(lc_messages)
                    elapsed = time.monotonic() - t0
                self._bump("calls")
                text = _message_text(response)
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
        messages: MessagesLike,
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

    # ------------------------------------------------------------------ LCEL
    def runnable(self, *, purpose: str, temperature: float = 0.0, json_mode: bool = False,
                 max_tokens: int | None = None) -> Runnable:
        """This client as an LCEL step: PromptValue | messages -> str, with the retry policy applied."""
        return RunnableLambda(
            lambda prompt: self.chat(prompt, temperature=temperature, json_mode=json_mode,
                                     max_tokens=max_tokens, purpose=purpose),
            name=f"llm:{purpose}",
        )

    def text_chain(self, prompt: BasePromptTemplate, *, purpose: str, temperature: float = 0.0,
                   max_tokens: int | None = None) -> Runnable:
        """``prompt | llm`` -> str. Raises LLMError when the model is unreachable."""
        return prompt | self.runnable(purpose=purpose, temperature=temperature, max_tokens=max_tokens)

    def json_chain(self, prompt: BasePromptTemplate, *, purpose: str, default: dict | None = None,
                   max_tokens: int | None = None) -> Runnable:
        """``prompt | llm(json) | parser`` -> dict, falling back to ``default`` (or {}) on an
        unreachable model or unparseable output — same contract as ``chat_json``."""
        def _fallback(_input) -> dict:
            logger.warning(f"LLM[{purpose}] produced no usable JSON — using the default")
            return dict(default or {})

        chain = prompt | self.runnable(purpose=purpose, json_mode=True, max_tokens=max_tokens) | JsonObjectOutputParser()
        return chain.with_fallbacks([RunnableLambda(_fallback, name=f"default:{purpose}")])

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


def _to_messages(messages: MessagesLike) -> list[BaseMessage]:
    if isinstance(messages, PromptValue):
        return messages.to_messages()
    return convert_to_messages(list(messages))


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content or [])


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
