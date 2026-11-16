"""Multi-query retriever: original + rewritten + alternative + HyDE variants,
each searched with the vector store's dense+sparse hybrid search, then fused with RRF.

It is a LangChain ``BaseRetriever``. The LangGraph agent calls
``search_variants()`` directly (it runs the rewrite / HyDE steps as separate
graph nodes so each is visible in the trace). Everything else (RAGSystem, the
retrieval evaluator) calls the standard ``invoke(query)``, which runs the same
techniques end to end, so the evaluation scripts measure the same retriever the
chatbot uses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from research_rag.models import Chunk
from .fusion import rrf_fuse

logger = logging.getLogger(__name__)


@dataclass
class QueryVariant:
    name: str            # "original" | "rewritten" | "alt1" | "hyde" ...
    text: str
    weight: float = 1.0


@dataclass
class RetrievalResult:
    documents: list[Document]
    variants_used: list[str] = field(default_factory=list)
    hits_per_variant: dict[str, int] = field(default_factory=dict)

    @property
    def chunks(self) -> list[Chunk]:
        return [Chunk.from_document(doc) for doc in self.documents]


class AdvancedRetriever(BaseRetriever):
    vector_store: Any                 # QdrantHybridVectorStore: .embeddings, .sparse_encoder, .hybrid_search_by_vector
    rewriter: Any = None              # QueryRewriter
    hyde: Any = None                  # HyDEGenerator
    use_rewrite: bool = True
    use_hyde: bool = True
    hyde_weight: float = 0.8          # HyDE is speculative text: it may add recall, but shouldn't outvote the real question
    rrf_k: int = 60
    k: int = 20

    # -- LangChain retriever contract (used by RAGSystem and the evaluators) --------
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                k: int | None = None) -> list[Document]:
        variants = [QueryVariant("original", query)]
        if self.use_rewrite and self.rewriter is not None:
            rewrite = self.rewriter.rewrite(query)
            variants.append(QueryVariant("rewritten", rewrite.standalone_question))
        if self.use_hyde and self.hyde is not None:
            passage = self.hyde.generate(variants[-1].text)
            if passage:
                variants.append(QueryVariant("hyde", passage, self.hyde_weight))
        return self.search_variants(variants, k or self.k).documents

    # -- core --------------------------------------------------------------------
    def search_variants(self, variants: list[QueryVariant], top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.k
        variants = _dedupe(variants)
        if not variants:
            return RetrievalResult([])

        logger.info(f"Multi-query retrieval over {len(variants)} variant(s): {[v.name for v in variants]}")
        dense_vectors = self.vector_store.embeddings.embed_documents([v.text for v in variants])  # one API call

        rankings, weights, hits, used = [], [], {}, []
        last_error: Exception | None = None
        for variant, dense in zip(variants, dense_vectors):
            try:
                sparse = self.vector_store.sparse_encoder.encode(variant.text)
                results = self.vector_store.hybrid_search_by_vector(dense, sparse, k=top_k)
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
        logger.info(f"Fused {sum(hits.values())} hits into {len(fused)} unique document(s)")
        return RetrievalResult(fused, used, hits)


def _dedupe(variants: list[QueryVariant]) -> list[QueryVariant]:
    seen, out = set(), []
    for v in variants:
        key = " ".join(v.text.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out
