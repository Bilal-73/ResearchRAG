from fakes import CORPUS, FakeEmbedder, FakeOpenAI, FakeVectorStore, make_chunk
from research_rag.llm import LLMClient
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
    emb, store = FakeEmbedder(), FakeVectorStore()
    r = AdvancedRetriever(emb, store)
    res = r.search_variants([QueryVariant("original", "kalman filter excitation"),
                             QueryVariant("rewritten", "Kalman  filter  excitation"),      # duplicate after normalisation
                             QueryVariant("hyde", "forgetting stability estimation")], top_k=5)
    assert len(emb.embed_calls) == 1 and len(emb.embed_calls[0]) == 2
    assert res.variants_used == ["original", "hyde"]
    assert res.chunks[0].id == "c3"


def test_hyde_adds_recall_for_vocabulary_mismatch():
    """The question shares no words with the answer chunk; the HyDE passage does."""
    emb, store = FakeEmbedder(), FakeVectorStore()
    r = AdvancedRetriever(emb, store)
    plain = r.search_variants([QueryVariant("original", "robustness")], top_k=3).chunks
    assert all(c.id != "c3" for c in plain)
    with_hyde = r.search_variants([QueryVariant("original", "robustness"),
                                   QueryVariant("hyde", "forgetting stability kalman")], top_k=3).chunks
    assert any(c.id == "c3" for c in with_hyde)


def test_one_failing_variant_does_not_sink_retrieval():
    class Flaky(FakeVectorStore):
        def hybrid_search(self, dense, sparse, limit=20, filters=None):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("boom")
            return super().hybrid_search(dense, sparse, limit, filters)
    r = AdvancedRetriever(FakeEmbedder(), Flaky())
    res = r.search_variants([QueryVariant("original", "impedance"), QueryVariant("rewritten", "impedance estimation")])
    assert res.variants_used == ["rewritten"] and res.chunks


def test_rewriter_falls_back_to_original_when_llm_is_useless():
    fake = FakeOpenAI(); fake.handlers["rewrite"] = lambda m: "garbage"
    rw = QueryRewriter(LLMClient(client=fake, deployment="d"))
    out = rw.rewrite("what about it?")
    assert out.standalone_question == "what about it?" and not out.used_llm


def test_hyde_failure_returns_empty_string():
    fake = FakeOpenAI(); fake.handlers["hyde"] = lambda m: RuntimeError("down")
    assert HyDEGenerator(LLMClient(client=fake, deployment="d", max_retries=0)).generate("q") == ""


def test_retrieve_convenience_path_used_by_evaluators():
    fake = FakeOpenAI()
    llm = LLMClient(client=fake, deployment="d")
    r = AdvancedRetriever(FakeEmbedder(), FakeVectorStore(), QueryRewriter(llm), HyDEGenerator(llm))
    assert r.retrieve("What adverse effects were reported?", top_k=3)
    assert fake.count("rewrite") == 1 and fake.count("hyde") == 1
