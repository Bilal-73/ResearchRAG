# generation/

Retrieved chunks -> grounded answer, with citations attached by the system.

- **`base.py`** — `BaseGenerator` interface.
- **`prompt_builder.py`** — `ANSWER_PROMPT`, a LangChain `ChatPromptTemplate` (answer only from
  context, numbered [S#] citations), and `CitationPromptBuilder`, which fills its variables and formats
  each excerpt as `[S#] paper > section, p.N` inside a fenced block.
- **`generator.py`** — `AnswerGenerator`: runs the `ANSWER_PROMPT | llm` chain, then builds `SourceCitation`
  objects from chunk **metadata** (never from anything the user typed) into the returned
  `RAGResponse`.

This is the only place citation logic exists. There is deliberately no parameter here
for a caller-supplied source or filter.
