"""Dense embeddings via Azure OpenAI, plus a simple hashing-based sparse encoder
for hybrid search. Swap sparse_encode's implementation for SPLADE/BM25-over-corpus
later — nothing outside this class needs to change."""
import hashlib
import logging
import re
import time
from collections import Counter

from research_rag.caching import RedisCache
from .base import BaseEmbedder

logger = logging.getLogger(__name__)

SPARSE_VOCAB_SIZE = 30_000
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class AzureEmbedder(BaseEmbedder):
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str,
                 cache: RedisCache | None = None):
        logger.debug(f"Initializing AzureEmbedder: endpoint={endpoint}, deployment={deployment}, "
                     f"api_version={api_version}, cache={'on' if cache and cache.enabled else 'off'}")
        from openai import AzureOpenAI

        self._client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self._deployment = deployment
        self._cache = cache

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("embed_batch called with an empty list — returning empty result")
            return []

        results: list[list[float] | None] = [None] * len(texts)
        cache_namespace = f"embed:{self._deployment}"
        to_fetch_indices: list[int] = []
        to_fetch_texts: list[str] = []

        if self._cache is not None:
            for i, text in enumerate(texts):
                cached = self._cache.get(cache_namespace, text)
                if cached is not None:
                    results[i] = cached
                else:
                    to_fetch_indices.append(i)
                    to_fetch_texts.append(text)
        else:
            to_fetch_indices = list(range(len(texts)))
            to_fetch_texts = texts

        if to_fetch_texts:
            logger.info(f"Embedding {len(to_fetch_texts)}/{len(texts)} text(s) via Azure OpenAI "
                        f"deployment={self._deployment} ({len(texts) - len(to_fetch_texts)} served from cache) ...")
            t0 = time.monotonic()
            response = self._client.embeddings.create(model=self._deployment, input=to_fetch_texts)
            elapsed = time.monotonic() - t0
            logger.info(f"Embedded {len(to_fetch_texts)} text(s) in {elapsed:.1f}s")

            for idx, item, text in zip(to_fetch_indices, response.data, to_fetch_texts):
                results[idx] = item.embedding
                if self._cache is not None:
                    self._cache.set(cache_namespace, text, item.embedding)
        else:
            logger.info(f"All {len(texts)} embedding(s) served from cache, 0 API calls")

        return results  # type: ignore[return-value]

    def sparse_encode(self, text: str) -> dict[str, list]:
        """Hashing vectorizer: token -> stable index in [0, SPARSE_VOCAB_SIZE), TF-weighted.
        Different tokens can hash into the same bucket (a collision) — Qdrant requires
        unique indices per point, so collisions are merged (counts summed) rather than
        left as duplicate index entries. Not cached — this is pure local computation,
        no API call to save."""
        tokens = _TOKEN_RE.findall(text.lower())
        counts = Counter(tokens)

        bucket_totals: dict[int, float] = {}
        for token, count in counts.items():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % SPARSE_VOCAB_SIZE
            bucket_totals[idx] = bucket_totals.get(idx, 0.0) + float(count)

        return {"indices": list(bucket_totals.keys()), "values": list(bucket_totals.values())}
