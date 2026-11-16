"""End-to-end RAG quality via RAGAS.

Two things about wiring RAGAS to this project that are easy to get wrong, both
verified against the installed version (ragas 0.3.9):

1. **It will not use your Azure setup on its own.** `ragas.evaluate()` reaches
   for a plain `OPENAI_API_KEY` unless you hand it judge models explicitly, so
   `build_ragas_models()` builds them from `config.settings` — no separate
   OpenAI key is needed anywhere in this project.

2. **The deprecated Langchain wrappers are the ones that work.** RAGAS 0.3.9
   deprecates `LangchainLLMWrapper` in favour of `llm_factory`, but the object
   `llm_factory` returns (`InstructorLLM`) has no `agenerate_prompt`, which is
   what the metrics call — every score comes back `nan`. So the wrappers stay,
   and their DeprecationWarning is silenced here rather than printed on every
   run. Revisit when RAGAS's own metrics support the factory path.

Column names also changed in 0.2+: samples are `user_input` / `response` /
`retrieved_contexts` / `reference`, not the old question/answer/contexts/
ground_truth, and the dataset is an `EvaluationDataset`, not a HF `Dataset`.

Metrics, and why these four:
  faithfulness       is every claim in the answer supported by the retrieved context?
  answer_relevancy   does the answer address the question actually asked?
  context_precision  are the retrieved chunks relevant, or padded with noise?
  context_recall     did retrieval find everything the reference answer needs?

The last two need a reference answer, which the benchmark supplies; the first
two do not, which is what makes them usable at inference time too (see
`InferenceScorer`).
"""
from __future__ import annotations

import logging
import warnings

from research_rag.guardrails import strip_annotations

logger = logging.getLogger(__name__)

# Metrics that need no reference answer — the only ones usable on a live turn.
REFERENCE_FREE_METRICS = ("faithfulness", "answer_relevancy")


def build_ragas_models(settings=None):
    """Judge LLM + judge embeddings, pointed at this project's Azure deployments."""
    from config.settings import get_settings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        from research_rag.llm import build_chat_model, build_embeddings

        settings = settings or get_settings()
        logger.info(f"RAGAS judge LLM -> Azure deployment={settings.azure_openai_chat_deployment}")
        # Same factories as the agent; RAGAS drives the model itself, so let the SDK retry.
        judge_llm = build_chat_model(settings, max_retries=2)
        judge_embeddings = build_embeddings(settings)
        return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(judge_embeddings)


def _metrics(names: list[str]):
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    available = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    return [available[n] for n in names if n in available]


def score_samples(samples: list[dict], metric_names: list[str], settings=None,
                  judge=None) -> list[dict]:
    """Score prepared samples. Each sample: user_input / response / retrieved_contexts [/ reference].

    Returns one dict of scores per sample, with `None` where RAGAS produced NaN
    (a metric it could not compute) — reporting a failed metric as 0.0 would
    look like a quality problem rather than a measurement problem.
    """
    import math

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    if not samples:
        return []

    judge_llm, judge_embeddings = judge or build_ragas_models(settings)
    dataset = EvaluationDataset(samples=[SingleTurnSample(**s) for s in samples])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = evaluate(dataset, metrics=_metrics(metric_names),
                          llm=judge_llm, embeddings=judge_embeddings, show_progress=False)

    rows: list[dict] = []
    for row in result.to_pandas().to_dict(orient="records"):
        scores = {}
        for name in metric_names:
            value = row.get(name)
            try:
                scores[name] = None if value is None or math.isnan(float(value)) else round(float(value), 4)
            except (TypeError, ValueError):
                scores[name] = None
        rows.append(scores)
    return rows


class RagasEvaluator:
    """Offline benchmark evaluation: runs the full agent per question, then scores."""

    METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def __init__(self, rag_system, settings=None):
        self.rag_system = rag_system
        self.settings = settings

    def evaluate(self, eval_items: list[dict], progress=None) -> dict:
        logger.info(f"RAGAS: running the full pipeline on {len(eval_items)} question(s)")

        samples, meta = [], []
        for i, item in enumerate(eval_items, start=1):
            result = self.rag_system.ask_full(item["question"])
            samples.append({
                "user_input": item["question"],
                "response": strip_annotations(result.answer),
                "retrieved_contexts": [c.text for c in result.chunks] or [""],
                "reference": item.get("expected_answer", "") or "",
            })
            meta.append({"query_id": item.get("query_id"), "question": item["question"],
                         "source": item.get("source", "unknown"), "type": item.get("type", "unknown"),
                         "answer": result.answer, "answer_kind": result.answer_kind})
            if progress:
                progress(i, len(eval_items), item["question"])

        logger.info(f"RAGAS: scoring {len(samples)} sample(s) on {', '.join(self.METRICS)}")
        scored = score_samples(samples, self.METRICS, self.settings)

        rows = [{**m, **s} for m, s in zip(meta, scored)]
        return self._summarise(rows)

    def _summarise(self, rows: list[dict]) -> dict:
        def mean(values):
            values = [v for v in values if v is not None]
            return round(sum(values) / len(values), 4) if values else None

        summary = {f"avg_{name}": mean([r.get(name) for r in rows]) for name in self.METRICS}
        summary["n_questions"] = len(rows)
        summary["n_scored"] = {name: sum(1 for r in rows if r.get(name) is not None) for name in self.METRICS}

        by_source: dict[str, list[dict]] = {}
        for row in rows:
            by_source.setdefault(row.get("source", "unknown"), []).append(row)
        summary["by_source"] = {
            source: {f"avg_{name}": mean([r.get(name) for r in group]) for name in self.METRICS}
                    | {"n": len(group)}
            for source, group in sorted(by_source.items())
        }
        summary["details"] = rows
        return summary


class InferenceScorer:
    """Scores a single live answer with the reference-free RAGAS metrics.

    Used by the API when a request opts in. Deliberately limited to
    faithfulness and answer_relevancy: the other two need a reference answer,
    which by definition does not exist for a real user's question.

    Built once and reused — constructing the judge models per request would add
    seconds to every turn.
    """

    def __init__(self, settings=None, metrics: list[str] | None = None):
        self.settings = settings
        self.metrics = list(metrics or REFERENCE_FREE_METRICS)
        self._judge = None

    def _ensure_judge(self):
        if self._judge is None:
            self._judge = build_ragas_models(self.settings)
        return self._judge

    def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        """Returns {metric: float|None} plus `error` if scoring could not run.

        Never raises: a judge failure must not turn a good answer into a failed
        request, so it degrades to nulls and says why.
        """
        if not question or not answer or not contexts:
            return {name: None for name in self.metrics} | {
                "error": "nothing to score (no answer or no retrieved context)"}
        try:
            scored = score_samples(
                [{"user_input": question, "response": answer, "retrieved_contexts": contexts}],
                self.metrics, self.settings, judge=self._ensure_judge(),
            )
        except Exception as error:  # noqa: BLE001 — scoring is an extra, never a dependency
            logger.warning(f"Inference-time RAGAS scoring failed: {error}")
            return {name: None for name in self.metrics} | {"error": str(error)[:200]}
        return scored[0] if scored else {name: None for name in self.metrics}
