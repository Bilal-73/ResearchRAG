"""Redis-backed cache for expensive, deterministic calls (embeddings, VLM
captions). The "context engineering" payoff: re-running ingestion or evaluation
after a code fix no longer re-embeds/re-captions everything from scratch —
only genuinely new text/images cost real API time.

Fails open: if Redis is unreachable, caching is silently disabled and every
call falls through to the real API — a missing cache should never break the
pipeline, only make it slower.

Instrumented on purpose. Cache behaviour is invisible from the outside, and
"is this actually saving me anything?" is only answerable with hit/miss
counters, so every operation is tallied and exposed through `stats()` /
`info()` (surfaced by `GET /v1/cache`). Counters are per-process and reset on
restart; the key counts come from Redis itself and are shared.
"""
import hashlib
import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Namespaces are prefixed "<prefix>:<namespace>:<sha256>", so a scan pattern per
# namespace is enough to count what is cached without reading any values.
NAMESPACE_LABELS = {
    "embed": "Embeddings (one per unique chunk/query text)",
    "caption": "Figure captions (one per figure image)",
}


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int = 60 * 60 * 24 * 30, prefix: str = "researchrag"):
        self._ttl = ttl_seconds
        self._prefix = prefix
        self._url = url
        self._client = None
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0,
                       "get_seconds": 0.0, "set_seconds": 0.0}

        try:
            import redis

            client = redis.from_url(url, socket_connect_timeout=2)
            client.ping()
            self._client = client
            logger.info(f"Redis cache connected: {url}")
        except Exception as error:  # noqa: BLE001 — any failure here just disables caching
            logger.warning(f"Redis unavailable ({error}) — caching disabled, calls will hit the real API")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def url(self) -> str:
        return self._url

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def _bump(self, key: str, amount=1) -> None:
        with self._lock:
            self._stats[key] += amount

    def _key(self, namespace: str, raw_key: str | bytes) -> str:
        if isinstance(raw_key, str):
            raw_key = raw_key.encode("utf-8")
        digest = hashlib.sha256(raw_key).hexdigest()
        return f"{self._prefix}:{namespace}:{digest}"

    def get(self, namespace: str, raw_key: str | bytes):
        if not self.enabled:
            return None
        t0 = time.monotonic()
        try:
            cached = self._client.get(self._key(namespace, raw_key))
        except Exception as error:  # noqa: BLE001
            self._bump("errors")
            logger.warning(f"Redis GET failed ({error}) — treating as cache miss")
            return None
        self._bump("get_seconds", time.monotonic() - t0)
        if cached is None:
            self._bump("misses")
            return None
        self._bump("hits")
        return json.loads(cached)

    def set(self, namespace: str, raw_key: str | bytes, value) -> None:
        if not self.enabled:
            return
        t0 = time.monotonic()
        try:
            self._client.setex(self._key(namespace, raw_key), self._ttl, json.dumps(value))
        except Exception as error:  # noqa: BLE001
            self._bump("errors")
            logger.warning(f"Redis SET failed ({error}) — continuing without caching this value")
            return
        self._bump("sets")
        self._bump("set_seconds", time.monotonic() - t0)

    # ---------------------------------------------------------------- introspection
    def stats(self) -> dict:
        """Per-process counters. `hit_rate` is over lookups only (hits + misses)."""
        with self._lock:
            snapshot = dict(self._stats)
        lookups = snapshot["hits"] + snapshot["misses"]
        snapshot["lookups"] = lookups
        snapshot["hit_rate"] = round(snapshot["hits"] / lookups, 4) if lookups else None
        snapshot["avg_get_ms"] = round(snapshot["get_seconds"] / lookups * 1000, 3) if lookups else None
        snapshot["avg_set_ms"] = (round(snapshot["set_seconds"] / snapshot["sets"] * 1000, 3)
                                  if snapshot["sets"] else None)
        for key in ("get_seconds", "set_seconds"):
            snapshot[key] = round(snapshot[key], 4)
        return snapshot

    def namespace_counts(self) -> dict[str, int]:
        """How many keys exist per namespace. Uses SCAN, never KEYS — KEYS blocks
        the whole server, which is fine on a dev box and a bad habit to build."""
        if not self.enabled:
            return {}
        counts: dict[str, int] = {}
        try:
            for namespace in NAMESPACE_LABELS:
                total = 0
                for _ in self._client.scan_iter(match=f"{self._prefix}:{namespace}:*", count=500):
                    total += 1
                counts[namespace] = total
        except Exception as error:  # noqa: BLE001
            self._bump("errors")
            logger.warning(f"Redis SCAN failed ({error})")
        return counts

    def server_info(self) -> dict:
        if not self.enabled:
            return {}
        try:
            info = self._client.info()
        except Exception as error:  # noqa: BLE001
            self._bump("errors")
            logger.warning(f"Redis INFO failed ({error})")
            return {}
        return {
            "version": info.get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "uptime_seconds": info.get("uptime_in_seconds"),
            "total_keys": sum(v.get("keys", 0) for k, v in info.items()
                              if k.startswith("db") and isinstance(v, dict)),
        }

    def info(self) -> dict:
        """Everything `GET /v1/cache` reports."""
        return {
            "enabled": self.enabled,
            "url": self._url,
            "ttl_seconds": self._ttl,
            "key_prefix": self._prefix,
            "stats": self.stats(),
            "namespaces": [
                {"name": name, "label": NAMESPACE_LABELS[name], "keys": count}
                for name, count in sorted(self.namespace_counts().items())
            ],
            "server": self.server_info(),
        }

    def reset_stats(self) -> None:
        """Zero the per-process counters (not the cached data) so a demo can show a
        clean cold-vs-warm comparison."""
        with self._lock:
            self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0,
                           "get_seconds": 0.0, "set_seconds": 0.0}
