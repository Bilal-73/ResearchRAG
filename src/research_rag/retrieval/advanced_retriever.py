"""Multi-query retriever: original + rewritten + alternative + HyDE variants,
each searched with the existing dense+sparse hybrid search, then fused with RRF.

The LangGraph agent calls `search_variants()` directly (it runs the rewrite /
HyDE steps as separate graph nodes so each is visible in the trace). Everything
else (RAGSystem.ask, the retrieval evaluator) calls plain `retrieve()`, which
runs the same techniques end to end, so the evaluation scripts measure the
same retriever the chatbot uses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from research_rag.embeddings.base import BaseEmbedder
from research_rag.models import Chunk
from research_rag.vectorstore.base import BaseVectorStore
from .base import BaseRetriever
from .fusion import rrf_fuse
from .query_transforms import HyDEGenerator, QueryRewriter

logger = logging.getLogger(__name__)


@dataclass
class QueryVariant:
    name: str            # "original" | "rewritten" | "alt1" | "hyde" ...
    text: str
    weight: float = 1.0


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    variants_used: list[str] = field(default_factory=list)
    hits_per_variant: dict[str, int] = field(default_factory=dict)


class AdvancedRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        rewriter: QueryRewriter | None = None,
        hyde: HyDEGenerator | None = None,
        use_rewrite: bool = True,
        use_hyde: bool = True,
        hyde_weight: float = 0.8,    # HyDE is speculative text: it may add recall, but shouldn't outvote the real question
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.rewriter = rewriter
        self.hyde = hyde
        self.use_rewrite = use_rewrite
        self.use_hyde = use_hyde
        self.hyde_weight = hyde_weight
        self.rrf_k = rrf_k

    # -- BaseRetriever contract (used by RAGSystem.ask and the evaluators) ---------
    def retrieve(self, query: str, top_k: int = 20) -> list[Chunk]:
        variants = [QueryVariant("original", query)]
        if self.use_rewrite and self.rewriter is not None:
            rewrite = self.rewriter.rewrite(query)
            variants.append(QueryVariant("rewritten", rewrite.standalone_question))
        if self.use_hyde and self.hyde is not None:
            passage = self.hyde.generate(variants[-1].text)
            if passage:
                variants.append(QueryVariant("hyde", passage, self.hyde_weight))
        return self.search_variants(variants, top_k).chunks

    # -- core --------------------------------------------------------------------
    def search_variants(self, variants: list[QueryVariant], top_k: int = 20) -> RetrievalResult:
        variants = _dedupe(variants)
        if not variants:
            return RetrievalResult([])

        logger.info(f"Multi-query retrieval over {len(variants)} variant(s): {[v.name for v in variants]}")
        dense_vectors = self.embedder.embed_batch([v.text for v in variants])   # one API call for all variants

        rankings, weights, hits, used = [], [], {}, []
        last_error: Exception | None = None
        for variant, dense in zip(variants, dense_vectors):
            try:
                sparse = self.embedder.sparse_encode(variant.text)
                results = self.vector_store.hybrid_search(dense, sparse, limit=top_k)
            except Exception as error:  # noqa: BLE001 — one failing variant shouldn't sink the others
                last_error = error
                logger.warning(f"Variant '{variant.name}' search failed: {error}")
                continue
            rankings.append(results)
            weights.append(variant.weight)
            hits[variant.name] = len(results)
            used.append(variant.name)

        if not rankings:
            raise RuntimeError(f"Every retrieval variant failed; last error: {last_error}") from last_error

        fused = rrf_fuse(rankings, weights, k=self.rrf_k)[:top_k]
        logger.info(f"Fused {sum(hits.values())} hits into {len(fused)} unique chunk(s)")
        return RetrievalResult(fused, used, hits)


def _dedupe(variants: list[QueryVariant]) -> list[QueryVariant]:
    seen, out = set(), []
    for v in variants:
        key = " ".join(v.text.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out
