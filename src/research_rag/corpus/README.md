# corpus/

`CorpusInfo` asks Qdrant which papers are actually indexed (cached 60s) and joins
`data/arxiv_manifest.json` for titles, authors and arXiv categories. Prompts, scope redirects and
`GET /v1/corpus` all read from it, so they stay accurate as you index more papers.

The corpus is too large to enumerate: `describe()` summarises it by field and category with a few
example titles, which is what the classifier and the out-of-scope redirect actually need in order
to judge "could this corpus plausibly cover the question?". Listing 400 paper titles into every
prompt would cost more than the answer.

If Qdrant goes away, the last known paper list is kept and `error` is set, so the assistant degrades
to a stale-but-sane scope description rather than claiming it has no corpus.
