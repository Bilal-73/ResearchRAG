# evaluation/

Scoring against the [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench):
its queries, its section-level relevance judgments, its reference answers. Nothing in this package
generates its own ground truth — that was the central weakness of the previous version, where
LLM-written questions were scored against the chunks they were written from.

- **`benchmark.py`** — builds the eval sets. `build_eval_set` keeps only queries whose gold
  document is actually indexed (otherwise you measure your ingestion budget, not your retriever)
  and samples **stratified across (type, source)** with a fixed seed, so table and image queries
  can't vanish into a text-heavy random sample. `build_negative_set` returns real benchmark
  queries whose gold document was deliberately left unindexed — genuinely unanswerable, but
  well-formed and on-topic, which is exactly what tempts a model to answer from memory.
- **`retrieval_metrics.py`** — `RetrievalEvaluator`: Recall@k, MRR, NDCG@k at **document** and
  **section** granularity, broken out by query type (abstractive/extractive) and modality
  (text / text-image / text-table / text-table-image). The section-level number is the strict one:
  right paper, wrong section counts as a miss, because the generator still won't find the fact.
  Note that this is not an LLM-free measurement by default: `AdvancedRetriever.invoke()` runs
  query rewriting and HyDE, so each query costs 2 LLM calls unless `ENABLE_QUERY_REWRITE` and
  `ENABLE_HYDE` are off. It is still far cheaper than a full run, which also pays generation,
  grading, groundedness and the judge.
- **`ragas_eval.py`** — **the default** answer-quality scorer, via
  [RAGAS](https://github.com/explodinggradients/ragas): `faithfulness`, `answer_relevancy`,
  `context_precision`, `context_recall`. Also hosts `InferenceScorer`, which scores a single live
  answer when an API request opts in. Two things worth knowing, both established against the
  installed version (ragas 0.3.9):
  - RAGAS will not use your Azure setup on its own — `build_ragas_models()` wires its judge LLM and
    embeddings to `config.settings`, so no separate OpenAI key is needed.
  - The *deprecated* `LangchainLLMWrapper` is the path that works. `llm_factory`, which RAGAS's own
    deprecation notice points at, returns an object with no `agenerate_prompt`; the metrics call
    that, so every score comes back `nan`. The warning is silenced rather than acted on, with the
    reason recorded in the module docstring.
- **`llm_judge.py`** — `LLMJudgeEvaluator`, the dependency-free alternative (`--judge builtin`):
  faithfulness, answer relevance, correctness (against the benchmark's reference answer) and
  citation quality. It scores two things RAGAS does not — correctness and citation quality — and
  needs nothing beyond `requirements.txt`, so it still runs when the RAGAS chain breaks.
  Refusals are counted separately rather than scored 0, so a system that games faithfulness by
  declining everything shows up immediately.
- **`refusal_eval.py`** — `RefusalEvaluator`: checks the system declines on the negative set.
  A refusal is detected both from the agent's own route (`not_found`) and the shared
  refusal-marker scan, since an answer can decline without using a stock phrase.

Everything is driven by `scripts/run_evaluation.py`, which records the full pipeline config
alongside the scores so two runs can be compared honestly.

## Offline vs inference-time scoring

The same RAGAS machinery runs in two places, with one important difference:

| | Offline (`run_evaluation.py`) | Inference (`POST /v1/chat` with `score: true`) |
|---|---|---|
| Questions | The benchmark's, with gold answers | Whatever a real user asked |
| Metrics | All four | **Reference-free only**: faithfulness, answer_relevancy |
| Why | `context_precision`/`context_recall` compare against a reference answer | A real question has no reference answer, so those two are not computable |

Inference scoring is opt-in per request and off by default: measured at ~15-20s of extra latency
per turn on gpt-4o-mini. The scorer is built once per process, on the first request that asks for
it, so only the first scored request pays to construct the judge models.

## Known limitations of this harness

**The negative set is "gold document not indexed", which is not the same as "unanswerable".**
As the corpus grows, topics overlap: a 50-paper run produced a question whose gold paper was
excluded but which another indexed paper genuinely answered, and the system was marked as
hallucinating for answering it correctly. Refusal accuracy is therefore a *lower bound*. Treat a
single failure as a case to read, not as a defect count.

**Over-refusal is invisible in faithfulness and silently dominates relevancy.** A refusal is
maximally faithful (it asserts nothing) but scores 0 on answer_relevancy, so a system that refuses
too often looks like one with a relevancy problem. Always read `avg_answer_relevancy` alongside the
share of answers containing "not reported in the retrieved context" — on one 60-query run, 12%
refusals moved relevancy from 0.85 to 0.74.

**RAGAS does not always score every row.** `context_precision` returned NaN on 11 of 60 rows in that
run. The averages here skip unscored rows rather than treating them as zero, and `n_scored` reports
how many actually contributed to each metric — check it before comparing two runs.

## Reading the numbers

- A large gap between `recall@k_doc` and `recall@k_section` means retrieval finds the right paper
  but the wrong part of it — usually a chunking problem, not an embedding one.
- A low `by_source` score for `text-table` / `text-image` against a healthy `text` score means the
  multimodal path is underperforming; try `CAPTION_FIGURES=true` and re-measure.
- High faithfulness with low correctness means the system is grounded in the wrong evidence —
  look at retrieval, not the generator.
- The judge shares a model family with the generator, which is a known source of self-preference
  bias. The retrieval metrics are judge-free; treat them as the more trustworthy signal.
