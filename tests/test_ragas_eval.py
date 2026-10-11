"""Tests for the RAGAS wiring.

RAGAS itself is faked here — these cover *our* logic around it, which is where
the mistakes live: NaN handling, reference-free metric selection, failure
degradation, and the summary maths. The one thing that genuinely needs the real
library (that `LangchainLLMWrapper` works and `llm_factory` does not) is
recorded in the module docstring, not asserted here, because it is a property of
the installed version rather than of this code.
"""
from __future__ import annotations

import pytest

from research_rag.evaluation import ragas_eval
from research_rag.evaluation.ragas_eval import InferenceScorer, RagasEvaluator


class FakeResult:
    """Stands in for ragas' EvaluationResult."""

    def __init__(self, records):
        self._records = records

    def to_pandas(self):
        records = self._records

        class _DF:
            @staticmethod
            def to_dict(orient="records"):
                return records

        return _DF()


@pytest.fixture
def fake_ragas(monkeypatch):
    """Intercepts score_samples' call into ragas, capturing what we sent it."""
    captured: dict = {}

    def fake_score_samples(samples, metric_names, settings=None, judge=None):
        captured["samples"] = samples
        captured["metrics"] = list(metric_names)
        return [{name: 0.9 for name in metric_names} for _ in samples]

    monkeypatch.setattr(ragas_eval, "score_samples", fake_score_samples)
    return captured


# ---------------------------------------------------------------- NaN handling --
def test_nan_scores_become_none_not_zero(monkeypatch):
    """RAGAS emits NaN for a metric it could not compute. Reporting that as 0.0
    would look like a quality failure instead of a measurement failure."""
    monkeypatch.setattr(ragas_eval, "build_ragas_models", lambda settings=None: ("llm", "emb"))
    monkeypatch.setattr(ragas_eval, "_metrics", lambda names: list(names))
    monkeypatch.setitem(
        __import__("sys").modules, "ragas",
        type("M", (), {"evaluate": staticmethod(
            lambda *a, **k: FakeResult([{"faithfulness": float("nan"), "answer_relevancy": 0.75}]))})(),
    )
    import sys
    schema = type("S", (), {})()
    schema.SingleTurnSample = lambda **kw: kw
    schema.EvaluationDataset = lambda samples: samples
    monkeypatch.setitem(sys.modules, "ragas.dataset_schema", schema)

    rows = ragas_eval.score_samples(
        [{"user_input": "q", "response": "a", "retrieved_contexts": ["c"]}],
        ["faithfulness", "answer_relevancy"],
    )
    assert rows[0]["faithfulness"] is None
    assert rows[0]["answer_relevancy"] == 0.75


# ---------------------------------------------------------- inference scoring --
def test_inference_scorer_uses_only_reference_free_metrics(fake_ragas):
    """context_precision / context_recall need a gold answer, which a live user
    question does not have — asking for them would score every real answer as NaN."""
    scorer = InferenceScorer()
    scorer._judge = ("llm", "emb")          # skip building real judge models
    scorer.score("What did it reach?", "It reached 2.34%.", ["the context"])

    assert fake_ragas["metrics"] == ["faithfulness", "answer_relevancy"]
    assert "reference" not in fake_ragas["samples"][0]


def test_inference_scorer_passes_question_answer_and_contexts_through(fake_ragas):
    scorer = InferenceScorer()
    scorer._judge = ("llm", "emb")
    scorer.score("the question", "the answer", ["ctx one", "ctx two"])

    sample = fake_ragas["samples"][0]
    assert sample["user_input"] == "the question"
    assert sample["response"] == "the answer"
    assert sample["retrieved_contexts"] == ["ctx one", "ctx two"]


@pytest.mark.parametrize("question,answer,contexts", [
    ("", "a", ["c"]),
    ("q", "", ["c"]),
    ("q", "a", []),
])
def test_nothing_to_score_is_reported_not_guessed(question, answer, contexts):
    result = InferenceScorer().score(question, answer, contexts)
    assert result["faithfulness"] is None and "nothing to score" in result["error"]


def test_a_judge_exception_degrades_to_nulls(monkeypatch):
    """Scoring is an extra. A judge outage must never turn a good answer into an error."""
    def boom(*args, **kwargs):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(ragas_eval, "score_samples", boom)
    scorer = InferenceScorer()
    scorer._judge = ("llm", "emb")
    result = scorer.score("q", "a", ["c"])
    assert result["faithfulness"] is None
    assert result["answer_relevancy"] is None
    assert "judge exploded" in result["error"]


def test_judge_models_are_built_once_and_reused(monkeypatch):
    calls = []
    monkeypatch.setattr(ragas_eval, "build_ragas_models",
                        lambda settings=None: calls.append(1) or ("llm", "emb"))
    monkeypatch.setattr(ragas_eval, "score_samples",
                        lambda *a, **k: [{"faithfulness": 1.0, "answer_relevancy": 1.0}])
    scorer = InferenceScorer()
    scorer.score("q", "a", ["c"])
    scorer.score("q2", "a2", ["c2"])
    assert len(calls) == 1          # constructing judge models per request would add seconds each time


# ------------------------------------------------------------ offline evaluator --
class FakeAgentResult:
    def __init__(self, answer, chunks, kind="rag"):
        self.answer = answer
        self.answer_kind = kind
        self.chunks = chunks


class FakeRagSystem:
    def __init__(self):
        from research_rag.models import Chunk, ChunkType

        self.chunk = Chunk(id="c1", text="the evidence", chunk_type=ChunkType.TEXT,
                           paper_id="2401.03345v2", section="Results", section_id=3)

    def ask_full(self, question):
        return FakeAgentResult(f"answer to {question}", [self.chunk])


def test_offline_evaluator_sends_the_reference_answer(fake_ragas):
    """Unlike inference, the benchmark HAS a gold answer, so the reference-dependent
    metrics are both available and worth paying for."""
    RagasEvaluator(FakeRagSystem()).evaluate(
        [{"query_id": "q1", "question": "What did it reach?", "expected_answer": "2.34%",
          "source": "text", "type": "extractive"}])

    assert fake_ragas["samples"][0]["reference"] == "2.34%"
    assert set(fake_ragas["metrics"]) == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"}


def test_offline_summary_averages_and_groups_by_source(fake_ragas):
    items = [
        {"query_id": "a", "question": "q1", "expected_answer": "x", "source": "text", "type": "extractive"},
        {"query_id": "b", "question": "q2", "expected_answer": "y", "source": "text-table", "type": "abstractive"},
    ]
    summary = RagasEvaluator(FakeRagSystem()).evaluate(items)

    assert summary["n_questions"] == 2
    assert summary["avg_faithfulness"] == 0.9
    assert summary["by_source"]["text"]["n"] == 1
    assert summary["by_source"]["text-table"]["n"] == 1
    assert len(summary["details"]) == 2


def test_summary_ignores_unscored_metrics_rather_than_counting_them_as_zero(monkeypatch):
    monkeypatch.setattr(
        ragas_eval, "score_samples",
        lambda samples, names, settings=None, judge=None: [
            {n: (None if n == "faithfulness" else 0.5) for n in names},
            {n: (1.0 if n == "faithfulness" else 0.5) for n in names},
        ])
    items = [
        {"query_id": "a", "question": "q1", "expected_answer": "x", "source": "text"},
        {"query_id": "b", "question": "q2", "expected_answer": "y", "source": "text"},
    ]
    summary = RagasEvaluator(FakeRagSystem()).evaluate(items)

    assert summary["avg_faithfulness"] == 1.0       # the single scored row, not (0 + 1) / 2
    assert summary["n_scored"]["faithfulness"] == 1
