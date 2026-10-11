from research_rag.models import Chunk, ChunkType
from research_rag.retrieval import HybridRetriever


class FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def sparse_encode(self, text):
        return {"indices": [1, 2], "values": [1.0, 1.0]}


class FakeVectorStore:
    def __init__(self, chunks_to_return):
        self._chunks_to_return = chunks_to_return
        self.last_filters = "NOT_CALLED"

    def hybrid_search(self, dense_vector, sparse_vector, limit=20, filters=None):
        self.last_filters = filters
        return self._chunks_to_return[:limit]


def _make_chunk(paper_id):
    return Chunk(
        id=f"{paper_id}-1", text="some text", chunk_type=ChunkType.TEXT,
        paper_id=paper_id, section="Results", page_no=3,
    )


def test_retrieve_returns_vector_store_results():
    chunks = [_make_chunk("PMC1"), _make_chunk("PMC2")]
    vector_store = FakeVectorStore(chunks)
    retriever = HybridRetriever(FakeEmbedder(), vector_store)

    results = retriever.retrieve("what was the HbA1c change?", top_k=2)

    assert results == chunks


def test_retrieve_never_passes_a_user_supplied_filter():
    """System decides relevance; there is no parameter here for the caller to
    inject a paper_id filter."""
    vector_store = FakeVectorStore([_make_chunk("PMC1")])
    retriever = HybridRetriever(FakeEmbedder(), vector_store)

    retriever.retrieve("any question", top_k=5)

    assert vector_store.last_filters is None
