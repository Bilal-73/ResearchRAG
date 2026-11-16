"""Cross-encoder reranking as a LangChain ``BaseDocumentCompressor`` — worth it here
because the precision of the top 3-5 chunks matters more than recall of the top 20
in a citation-driven system.

The relevance score is written onto each returned document
(``metadata["relevance_score"]``, surfaced as ``Chunk.score``) so the UI and the
citation list can show *why* a source ranked where it did.

``compress_documents`` keeps the standard compressor contract (``top_n``) so the
reranker composes with any LangChain retriever; ``rerank(query, docs, top_k)``
is the same operation with a per-call cut-off, which the agent needs.
"""
import logging
import time
from typing import Any, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class BaseReranker(BaseDocumentCompressor):
    top_n: int = 5

    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Callbacks | None = None) -> Sequence[Document]:
        return self.rerank(query, list(documents), top_k=self.top_n)

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        logger.debug(f"Loading cross-encoder model: {self.model_name} (first run downloads it from Hugging Face)")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)
        logger.debug("Cross-encoder loaded.")

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        if not documents:
            logger.warning("rerank called with 0 candidate documents — nothing to rerank")
            return []

        logger.info(f"Reranking {len(documents)} candidate(s) down to top_k={top_k} ...")
        t0 = time.monotonic()

        scores = self._model.predict([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
        for score, doc in ranked:
            doc.metadata["relevance_score"] = float(score)

        logger.info(f"Reranked in {time.monotonic() - t0:.2f}s")
        return [doc for _, doc in ranked][:top_k]


class NoOpReranker(BaseReranker):
    """Keeps the retrieval order (used when reranking is switched off, so the
    rest of the pipeline never needs a special case)."""

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        return list(documents)[:top_k]
