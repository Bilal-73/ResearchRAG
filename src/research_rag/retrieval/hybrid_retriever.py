"""Dense + sparse hybrid retrieval as a LangChain retriever. Relevance is decided
by the system only — no user-supplied paper_id or section filter is ever accepted here."""
import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    vector_store: Any          # QdrantHybridVectorStore (or any store with the same similarity_search)
    k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                k: int | None = None) -> list[Document]:
        k = k or self.k
        logger.info(f"Retrieving k={k} for query: {query!r}")
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} document(s)")
        return results
