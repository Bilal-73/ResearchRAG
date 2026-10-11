"""Dense + sparse hybrid retrieval. Relevance is decided by the system only —
no user-supplied paper_id or section filter is ever accepted here."""
import logging

from research_rag.embeddings.base import BaseEmbedder
from research_rag.models import Chunk
from research_rag.vectorstore.base import BaseVectorStore
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 20) -> list[Chunk]:
        logger.info(f"Retrieving top_k={top_k} for query: {query!r}")
        dense_vector = self.embedder.embed(query)
        sparse_vector = self.embedder.sparse_encode(query)
        results = self.vector_store.hybrid_search(dense_vector, sparse_vector, limit=top_k)
        logger.info(f"Retrieved {len(results)} chunk(s)")
        return results
