from langchain_core.documents import Document

from fakes import CORPUS, FakeEmbedder, FakeChatModel, FakeVectorStore, make_chunk
from research_rag.llm import LLMClient
from research_rag.models import Chunk
from research_rag.retrieval import AdvancedRetriever, HyDEGenerator, QueryRewriter, QueryVariant, rrf_fuse


def test_rrf_rewards_agreement_across_lists():
    a, b, c = make_chunk("a", "a"), make_chunk("b", "b"), make_chunk("c", "c")
    fused = rrf_fuse([[a, b, c], [c, b]])
    assert {x.id for x in fused[:2]} == {"b", "c"}            # chunks both lists agree on outrank the one only one list has
    assert fused[-1].id == "a"
    assert {x.id for x in fused} == {"a", "b", "c"}           # deduplicated


def test_rrf_weights_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        rrf_fuse([[make_chunk("a", "a")]], weights=[1, 2])


def test_search_variants_embeds_in_one_batch_and_dedupes():
    emb = FakeEmbedder()
    r = AdvancedRetriever(vector_store=FakeVectorStore(embedding=emb))
    res = r.search_variants([QueryVariant("original", "kalman filter excitation"),
                             QueryVariant("rewritten", "Kalman  filter  excitation"),      # duplicate after normalisation
                             QueryVariant("hyde", "forgetting stability estimation")], top_k=5)
    assert len(emb.embed_calls) == 1 and len(emb.embed_calls[0]) == 2
    assert res.variants_used == ["original", "hyde"]
    assert res.chunks[0].id == "c3"


def test_hyde_adds_recall_for_vocabulary_mismatch():
    """The question shares no words with the answer chunk; the HyDE passage does."""
    r = AdvancedRetriever(vector_store=FakeVectorStore())
    plain = r.search_variants([QueryVariant("original", "robustness")], top_k=3).chunks
    assert all(c.id != "c3" for c in plain)
    with_hyde = r.search_variants([QueryVariant("original", "robustness"),
                                   QueryVariant("hyde", "forgetting stability kalman")], top_k=3).chunks
    assert any(c.id == "c3" for c in with_hyde)


def test_one_failing_variant_does_not_sink_retrieval():
    class Flaky(FakeVectorStore):
        def hybrid_search_by_vector(self, dense, sparse, k=20):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("boom")
            return super().hybrid_search_by_vector(dense, sparse, k)
    r = AdvancedRetriever(vector_store=Flaky())
    res = r.search_variants([QueryVariant("original", "impedance"), QueryVariant("rewritten", "impedance estimation")])
    assert res.variants_used == ["rewritten"] and res.chunks


def test_rewriter_falls_back_to_original_when_llm_is_useless():
    fake = FakeChatModel(); fake.handlers["rewrite"] = lambda m: "garbage"
    rw = QueryRewriter(LLMClient(fake))
    out = rw.rewrite("what about it?")
    assert out.standalone_question == "what about it?" and not out.used_llm


def test_hyde_failure_returns_empty_string():
    fake = FakeChatModel(); fake.handlers["hyde"] = lambda m: RuntimeError("down")
    assert HyDEGenerator(LLMClient(fake, max_retries=0)).generate("q") == ""


def test_retrieve_convenience_path_used_by_evaluators():
    fake = FakeChatModel()
    llm = LLMClient(fake)
    r = AdvancedRetriever(vector_store=FakeVectorStore(), rewriter=QueryRewriter(llm), hyde=HyDEGenerator(llm))
    docs = r.invoke("What adverse effects were reported?", k=3)
    assert docs and len(docs) <= 3 and all(isinstance(d, Document) and d.id for d in docs)
    assert fake.count("rewrite") == 1 and fake.count("hyde") == 1


def test_rrf_fuses_langchain_documents_by_id():
    a, b = Document(id="a", page_content="x"), Document(id="b", page_content="y")
    fused = rrf_fuse([[a, b], [Document(id="b", page_content="y")]])
    assert [d.id for d in fused] == ["b", "a"]


def test_chunk_document_round_trip_keeps_payload_and_score():
    chunk = CORPUS[3]
    doc = chunk.to_document()
    assert doc.page_content == chunk.text and doc.metadata["figure_no"] == "I" and "text" not in doc.metadata
    doc.metadata["relevance_score"] = 0.7
    back = Chunk.from_document(doc)
    assert back.to_payload() == chunk.to_payload() and back.id == chunk.id and back.score == 0.7
