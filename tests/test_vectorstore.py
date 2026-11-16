"""The real QdrantHybridVectorStore against qdrant-client's in-process local mode
(no server): payload layout, hybrid search, and the LangChain VectorStore surface."""
import pytest

from fakes import CORPUS, FakeEmbedder, FakeSparseEncoder
from langchain_core.vectorstores import VectorStore

from research_rag.models import Chunk
from research_rag.retrieval import AdvancedRetriever, HybridRetriever, QueryVariant
from research_rag.vectorstore import QdrantHybridVectorStore

qdrant_client = pytest.importorskip("qdrant_client")


@pytest.fixture
def store():
    s = QdrantHybridVectorStore(qdrant_client.QdrantClient(":memory:"), "test", FakeEmbedder(), FakeSparseEncoder())
    s.create_collection()                     # dimension probed through the embeddings
    s.add_documents([c.to_document() for c in CORPUS])
    return s


def test_is_a_langchain_vector_store_with_flat_payload(store):
    assert isinstance(store, VectorStore)
    point = store._client.scroll("test", limit=1, with_payload=True)[0][0]
    assert {"text", "paper_id", "section", "section_id", "chunk_type"} <= set(point.payload)
    assert "metadata" not in point.payload and "chunk_id" not in point.payload   # same layout as the live collection


def test_hybrid_similarity_search_returns_documents_that_map_back_to_chunks(store):
    docs = store.similarity_search("kalman filter without excitation", k=3)
    assert docs and all(d.id and d.metadata["chunk_id"] == d.id for d in docs)
    chunks = [Chunk.from_document(d) for d in docs]
    assert chunks[0].text == CORPUS[2].text and chunks[0].section_id == 9


def test_reingesting_overwrites_instead_of_duplicating(store):
    store.add_documents([CORPUS[0].to_document()])
    assert store.count_chunks() == len(CORPUS)
    assert store.list_paper_ids() == ["2410.14077v2"]
    assert len(store.get_by_paper("2410.14077v2")) == len(CORPUS)


def test_retrievers_run_on_the_real_store(store):
    assert HybridRetriever(vector_store=store, k=2).invoke("grid impedance estimation")
    result = AdvancedRetriever(vector_store=store).search_variants(
        [QueryVariant("original", "rmspe excitation"), QueryVariant("hyde", "forgetting stability")], top_k=4)
    assert result.variants_used == ["original", "hyde"] and len(result.documents) <= 4
