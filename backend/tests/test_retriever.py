from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from research_rag.retrieval import HybridRetriever


class FakeVectorStore:
    def __init__(self, docs_to_return):
        self._docs_to_return = docs_to_return
        self.last_kwargs = "NOT_CALLED"

    def similarity_search(self, query, k=4, **kwargs):
        self.last_kwargs = kwargs
        return self._docs_to_return[:k]


def _make_doc(paper_id):
    return Document(id=f"{paper_id}-1", page_content="some text",
                    metadata={"chunk_type": "text", "paper_id": paper_id, "section": "Results", "page_no": 3})


def test_retrieve_returns_vector_store_results():
    docs = [_make_doc("PMC1"), _make_doc("PMC2")]
    retriever = HybridRetriever(vector_store=FakeVectorStore(docs))

    assert isinstance(retriever, BaseRetriever)
    assert retriever.invoke("what was the HbA1c change?", k=2) == docs


def test_retrieve_never_passes_a_user_supplied_filter():
    """System decides relevance; there is no parameter here for the caller to
    inject a paper_id filter."""
    vector_store = FakeVectorStore([_make_doc("PMC1")])
    retriever = HybridRetriever(vector_store=vector_store, k=5)

    retriever.invoke("any question")

    assert vector_store.last_kwargs == {}
