# generation/

Retrieved chunks -> grounded answer, with citations attached by the system.

- **`base.py`** — `BaseGenerator` interface.
- **`prompt_builder.py`** — `CitationPromptBuilder`: builds the system prompt (answer
  only from context, numbered [S#] citations) and formats each
  excerpt as `[S#] paper > section, p.N` inside a fenced block.
- **`generator.py`** — `AnswerGenerator`: calls the LLM, then builds `SourceCitation`
  objects from chunk **metadata** (never from anything the user typed) into the returned
  `RAGResponse`.

This is the only place citation logic exists. There is deliberately no parameter here
for a caller-supplied source or filter.
