"""Per-session + global request rate limiting (protects the app and the Azure quota).

Sliding-window log: each key keeps the timestamps of its recent requests, and a
request is admitted only if EVERY configured window still has room. Compared
with a fixed window this has no burst-at-the-boundary loophole (10/min can't
turn into 20 requests across a minute boundary).

In-memory and thread-safe — right for a single API process. If you run several
workers or replicas, swap the storage for Redis (INCR + EXPIRE); the interface
(`check_and_consume` / `usage`) is the only thing callers depend on.

Keyed on the API client (its key, or its IP when unauthenticated), never on the
conversation — otherwise a caller could reset its own quota by starting a new
conversation on every request.

Not to be confused with upstream 429 handling (retry + backoff), which is in
``llm/client.py``.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

GLOBAL_KEY = "__global__"


@dataclass(frozen=True)
class RateLimit:
    max_requests: int
    window_seconds: float
    name: str                      # shown in messages, e.g. "per minute"


@dataclass
class RateDecision:
    allowed: bool
    retry_after_seconds: float = 0.0
    limit_name: str = ""
    remaining: int | None = None   # requests left in the tightest window

    def message(self) -> str:
        wait = max(1, int(self.retry_after_seconds + 0.999))
        return (f"You're sending messages faster than the limit allows ({self.limit_name}). "
                f"Please wait about {wait} second{'s' if wait != 1 else ''} and try again.")


class SlidingWindowRateLimiter:
    def __init__(
        self,
        per_session: list[RateLimit],
        global_limits: list[RateLimit] | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_tracked_sessions: int = 5000,
    ):
        self._per_session = per_session
        self._global = global_limits or []
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._max_tracked = max_tracked_sessions
        self._horizon = max([rl.window_seconds for rl in per_session + self._global] or [0])

    def check_and_consume(self, session_id: str) -> RateDecision:
        now = self._clock()
        with self._lock:
            self._evict_stale(now)
            session_hits = self._prune(self._hits.setdefault(session_id, deque()), now)
            global_hits = self._prune(self._hits.setdefault(GLOBAL_KEY, deque()), now)

            for limit, hits in [(l, session_hits) for l in self._per_session] + \
                               [(l, global_hits) for l in self._global]:
                in_window = [t for t in hits if now - t < limit.window_seconds]
                if len(in_window) >= limit.max_requests:
                    retry_after = limit.window_seconds - (now - in_window[0])
                    return RateDecision(False, retry_after, limit.name, remaining=0)

            session_hits.append(now)
            global_hits.append(now)
            return RateDecision(True, remaining=self._remaining(session_hits, now))

    def usage(self, session_id: str) -> dict:
        """Read-only snapshot for the sidebar (does not consume quota)."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(session_id, deque())
            return {
                limit.name: {
                    "used": sum(1 for t in hits if now - t < limit.window_seconds),
                    "limit": limit.max_requests,
                }
                for limit in self._per_session
            }

    # -- internals -------------------------------------------------------------
    def _prune(self, hits: deque, now: float) -> deque:
        while hits and now - hits[0] >= self._horizon:
            hits.popleft()
        return hits

    def _remaining(self, hits: deque, now: float) -> int:
        return min(
            (limit.max_requests - sum(1 for t in hits if now - t < limit.window_seconds)
             for limit in self._per_session),
            default=0,
        )

    def _evict_stale(self, now: float) -> None:
        if len(self._hits) <= self._max_tracked:
            return
        for key in [k for k, h in self._hits.items() if k != GLOBAL_KEY and (not h or now - h[-1] >= self._horizon)]:
            del self._hits[key]


def build_rate_limiter(settings) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(
        per_session=[
            RateLimit(settings.rate_limit_per_minute, 60, "per minute"),
            RateLimit(settings.rate_limit_per_hour, 3600, "per hour"),
        ],
        global_limits=[RateLimit(settings.global_rate_limit_per_minute, 60, "server-wide, per minute")],
    )
