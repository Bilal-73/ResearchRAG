"""Evaluation-harness tests.

The point of these is that an evaluation you can't trust is worse than none:
if the eval set silently includes queries whose gold document was never
indexed, or scores a document-level hit as a section-level one, every number
downstream is wrong in a flattering direction. These lock that behaviour down.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.retrievers import BaseRetriever

from fakes import CORPUS, Harness, make_chunk
from research_rag.evaluation import (
    LLMJudgeEvaluator, RefusalEvaluator, RetrievalEvaluator, build_eval_set, build_negative_set,
)
from research_rag.rag_system import RAGSystem


class FakeCorpus:
    """Stands in for RAGBenchCorpus with a controllable query/qrel set."""

    def __init__(self, n_text=12, n_table=4, n_image=4, indexed_doc="2410.14077v2", other_doc="2401.07294v4"):
        self._queries, self._qrels, self._answers = {}, {}, {}
        spec = [("text", "abstractive", n_text), ("text-table", "extractive", n_table),
                ("text-image", "abstractive", n_image)]
        i = 0
        for source, qtype, count in spec:
            for _ in range(count):
                qid = f"q{i:03d}"
                doc = indexed_doc if i % 2 == 0 else other_doc      # half point at an unindexed paper
                self._queries[qid] = {"query": f"question {i} about {source}", "type": qtype, "source": source}
                self._qrels[qid] = {"doc_id": doc, "section_id": 8 if i % 3 else 9}
                self._answers[qid] = f"reference answer {i}"
                i += 1

    def queries(self): return self._queries
    def qrels(self): return self._qrels
    def answers(self): return self._answers


# ---------------------------------------------------------------- eval set -----
def test_eval_set_excludes_queries_whose_gold_document_is_not_indexed():
    eval_set = build_eval_set(FakeCorpus(), ["2410.14077v2"], n=None)
    assert eval_set.items and eval_set.n_dropped_unindexed > 0
    assert all(item["doc_id"] == "2410.14077v2" for item in eval_set.items)


def test_eval_set_carries_everything_the_scorers_need():
    item = build_eval_set(FakeCorpus(), ["2410.14077v2"], n=None).items[0]
    assert set(item) >= {"query_id", "question", "type", "source", "doc_id", "section_id", "expected_answer"}


def test_sampling_is_stratified_across_modalities_and_reproducible():
    corpus, indexed = FakeCorpus(n_text=40, n_table=20, n_image=20), ["2410.14077v2"]
    first = build_eval_set(corpus, indexed, n=10, seed=7)
    second = build_eval_set(corpus, indexed, n=10, seed=7)
    assert [i["query_id"] for i in first.items] == [i["query_id"] for i in second.items]
    assert len(first.items) == 10
    # every modality present in the pool survives sampling
    assert len(first.summary()["by_source"]) >= 2
    assert build_eval_set(corpus, indexed, n=10, seed=8).items != first.items


def test_filters_narrow_to_one_modality():
    eval_set = build_eval_set(FakeCorpus(), ["2410.14077v2"], n=None, sources=["text-table"])
    assert eval_set.items and all(i["source"] == "text-table" for i in eval_set.items)


def test_requesting_more_than_available_returns_everything():
    eval_set = build_eval_set(FakeCorpus(n_text=4, n_table=0, n_image=0), ["2410.14077v2"], n=999)
    assert len(eval_set.items) == eval_set.summary()["n_selected"] <= 4


def test_negative_set_is_only_queries_about_unindexed_documents():
    negatives = build_negative_set(FakeCorpus(), ["2410.14077v2"], n=5)
    assert len(negatives) == 5
    assert all(item["unindexed_doc_id"] != "2410.14077v2" for item in negatives)


def test_eval_and_negative_sets_never_overlap():
    corpus, indexed = FakeCorpus(), ["2410.14077v2"]
    answerable = {i["query_id"] for i in build_eval_set(corpus, indexed, n=None).items}
    unanswerable = {i["query_id"] for i in build_negative_set(corpus, indexed, n=50)}
    assert answerable and unanswerable and not (answerable & unanswerable)


# ---------------------------------------------------------------- retrieval ----
class ScriptedRetriever(BaseRetriever):
    chunks: list

    def _get_relevant_documents(self, query, *, run_manager, k=10):
        return [c.to_document() for c in self.chunks[:k]]


def test_section_level_scoring_is_stricter_than_document_level():
    # Right paper, wrong section: a document-level metric counts it, a section-level one must not.
    retriever = ScriptedRetriever(chunks=[make_chunk("x", "text", section_id=99)])
    result = RetrievalEvaluator(retriever, top_k=5).evaluate(
        [{"query_id": "q1", "question": "q", "doc_id": "2410.14077v2", "section_id": 8,
          "type": "extractive", "source": "text"}])
    assert result["overall"]["recall@k_doc"] == 1.0
    assert result["overall"]["recall@k_section"] == 0.0


def test_rank_sensitive_metrics_reward_a_higher_hit():
    miss = make_chunk("m", "irrelevant", section_id=1)
    hit = make_chunk("h", "relevant", section_id=8)
    item = [{"query_id": "q", "question": "q", "doc_id": "2410.14077v2", "section_id": 8,
             "type": "extractive", "source": "text"}]
    top = RetrievalEvaluator(ScriptedRetriever(chunks=[hit, miss]), top_k=5).evaluate(item)["overall"]
    second = RetrievalEvaluator(ScriptedRetriever(chunks=[miss, hit]), top_k=5).evaluate(item)["overall"]
    assert top["mrr_section"] == 1.0 and second["mrr_section"] == 0.5
    assert top["ndcg@k_section"] > second["ndcg@k_section"]


def test_metrics_are_broken_out_by_type_and_source():
    retriever = ScriptedRetriever(chunks=[make_chunk("h", "relevant", section_id=8)])
    items = [
        {"query_id": "a", "question": "q", "doc_id": "2410.14077v2", "section_id": 8,
         "type": "extractive", "source": "text-table"},
        {"query_id": "b", "question": "q", "doc_id": "2410.14077v2", "section_id": 42,
         "type": "abstractive", "source": "text"},
    ]
    result = RetrievalEvaluator(retriever, top_k=5).evaluate(items)
    assert result["by_source"]["text-table"]["recall@k_section"] == 1.0
    assert result["by_source"]["text"]["recall@k_section"] == 0.0
    assert result["by_type"]["extractive"]["n"] == 1


def test_a_total_miss_scores_zero_not_an_error():
    result = RetrievalEvaluator(ScriptedRetriever(chunks=[]), top_k=5).evaluate(
        [{"query_id": "q", "question": "q", "doc_id": "nope", "section_id": 1,
          "type": "extractive", "source": "text"}])
    assert result["overall"]["recall@k_doc"] == 0.0 and result["overall"]["mrr_doc"] == 0.0


def test_retrieval_evaluator_runs_against_the_real_retriever():
    h = Harness()
    result = RetrievalEvaluator(h.retriever, top_k=5, reranker=h.reranker).evaluate(
        [{"query_id": "q1", "question": "What RMSPE did the proposed algorithm reach?",
          "doc_id": "2410.14077v2", "section_id": 8, "type": "extractive", "source": "text"}])
    assert result["overall"]["recall@k_doc"] == 1.0


# ---------------------------------------------------------------- judge --------
def _system(harness: Harness) -> RAGSystem:
    return RAGSystem(harness.agent, harness.retriever, harness.reranker)


def test_judge_scores_every_dimension_and_summarises_by_source():
    h = Harness()
    items = [{"query_id": "q1", "question": "What RMSPE did the proposed algorithm reach?",
              "expected_answer": "2.3426 percent", "source": "text-table"}]
    result = LLMJudgeEvaluator(_system(h), h.llm).evaluate(items)
    assert result["avg_faithfulness"] == 1.0 and result["avg_citation_quality"] == 1.0
    assert result["n_questions"] == 1
    assert result["by_source"]["text-table"]["avg_correctness"] == 1.0
    assert result["details"][0]["n_cited_sources"] == 1


def test_judge_reports_refusals_on_answerable_questions():
    h = Harness()
    h.model.handlers["generate"] = lambda m: "This is not reported in the retrieved context."
    result = LLMJudgeEvaluator(_system(h), h.llm).evaluate(
        [{"query_id": "q1", "question": "What RMSPE did it reach?", "expected_answer": "2.3426", "source": "text"}])
    assert result["refusal_rate_on_answerable"] == 1.0


def test_unusable_judge_output_yields_none_not_a_crash_or_a_zero():
    h = Harness()
    h.model.handlers["judge"] = lambda m: "not json at all"
    result = LLMJudgeEvaluator(_system(h), h.llm).evaluate(
        [{"query_id": "q1", "question": "What RMSPE did it reach?", "expected_answer": "x", "source": "text"}])
    assert result["avg_faithfulness"] is None
    assert result["details"][0]["faithfulness"] is None


def test_evaluation_questions_do_not_share_conversation_memory():
    h = Harness()
    system = _system(h)
    system.ask_full("What RMSPE did the proposed algorithm reach?")
    second = system.ask_full("what about without excitation?")
    assert second.intent == "new_question"      # no history leaked from the previous question


# ---------------------------------------------------------------- refusal ------
def test_refusal_evaluator_counts_a_decline_as_correct():
    h = Harness(chunks=[])                      # nothing indexed -> the agent must decline
    result = RefusalEvaluator(_system(h)).evaluate(
        [{"query_id": "n1", "question": "What does the unindexed paper report?",
          "unindexed_doc_id": "2401.07294v4"}])
    assert result["refusal_accuracy"] == 1.0 and not result["hallucinations"]


def test_refusal_evaluator_flags_a_confident_answer_as_a_hallucination():
    h = Harness(chunks=CORPUS)
    h.model.handlers["generate"] = lambda m: "The unindexed paper reports an RMSPE of 1.1 percent [S1]."
    result = RefusalEvaluator(_system(h)).evaluate(
        [{"query_id": "n1", "question": "What does the unindexed paper report?",
          "unindexed_doc_id": "2401.07294v4"}])
    assert result["refusal_accuracy"] == 0.0
    assert result["hallucinations"][0]["question"].startswith("What does the unindexed")
