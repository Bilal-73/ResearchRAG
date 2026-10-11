"""Cross-encoder reranking pass — worth it here because the precision of the
top 3-5 chunks matters more than recall of the top 20 in a citation-driven system.

The relevance score is written onto each returned chunk (`chunk.score`) so the
UI and the citation list can show *why* a source ranked where it did."""
import logging
import time

from research_rag.models import Chunk
from .base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.debug(f"Loading cross-encoder model: {model_name} (first run downloads it from Hugging Face)")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        logger.debug("Cross-encoder loaded.")

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        if not chunks:
            logger.warning("rerank called with 0 candidate chunks — nothing to rerank")
            return []

        logger.info(f"Reranking {len(chunks)} candidate(s) down to top_k={top_k} ...")
        t0 = time.monotonic()

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        for score, chunk in ranked:
            chunk.score = float(score)

        logger.info(f"Reranked in {time.monotonic() - t0:.2f}s")
        return [chunk for _, chunk in ranked][:top_k]


class NoOpReranker(BaseReranker):
    """Keeps the retrieval order (used when reranking is switched off, so the
    rest of the pipeline never needs a special case)."""

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        return chunks[:top_k]
