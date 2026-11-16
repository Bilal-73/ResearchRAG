"""Redis-cached LangChain ``Embeddings``.

Wraps any LangChain embedding model (``AzureOpenAIEmbeddings`` in production)
so re-embedding the same text — a repeated query, a re-run ingestion — costs
no API call. The cache namespace is ``embed:<deployment>``, unchanged from the
pre-LangChain embedder, so existing cache entries stay valid.
"""
import logging
import time

from langchain_core.embeddings import Embeddings

from research_rag.caching import RedisCache

logger = logging.getLogger(__name__)


class CachedEmbeddings(Embeddings):
    def __init__(self, underlying: Embeddings, cache: RedisCache | None = None, namespace: str = "embed"):
        logger.debug(f"Initializing CachedEmbeddings: namespace={namespace}, "
                     f"cache={'on' if cache and cache.enabled else 'off'}")
        self.underlying = underlying
        self._cache = cache
        self._namespace = namespace

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("embed_documents called with an empty list — returning empty result")
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_fetch_indices: list[int] = []
        to_fetch_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(self._namespace, text) if self._cache is not None else None
            if cached is not None:
                results[i] = cached
            else:
                to_fetch_indices.append(i)
                to_fetch_texts.append(text)

        if to_fetch_texts:
            logger.info(f"Embedding {len(to_fetch_texts)}/{len(texts)} text(s) via {type(self.underlying).__name__} "
                        f"({len(texts) - len(to_fetch_texts)} served from cache) ...")
            t0 = time.monotonic()
            vectors = self.underlying.embed_documents(to_fetch_texts)
            logger.info(f"Embedded {len(to_fetch_texts)} text(s) in {time.monotonic() - t0:.1f}s")

            for idx, vector, text in zip(to_fetch_indices, vectors, to_fetch_texts):
                results[idx] = vector
                if self._cache is not None:
                    self._cache.set(self._namespace, text, vector)
        else:
            logger.info(f"All {len(texts)} embedding(s) served from cache, 0 API calls")

        return results  # type: ignore[return-value]
