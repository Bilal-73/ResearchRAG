"""Deterministic replies for routes that must not depend on an LLM: they are
instant, free, and can't be argued out of their wording."""

OUT_OF_SCOPE = (
    "That's outside what I can help with. I answer questions about {scope}, and every answer comes with "
    "citations to the exact paper and section.\n\nFor example: *\"What problem does this method solve and how "
    "is it evaluated?\"* or *\"Which baseline did the proposed approach outperform, and by how much?\"*"
)

NOT_FOUND = (
    "This is not reported in the retrieved context — I couldn't find anything relevant in {scope}."
)

CHITCHAT_FALLBACK = (
    "Hello! I answer questions about {scope}, with citations to the exact paper and section. "
    "Try asking about a method, a dataset, an experimental result, or a paper's limitations."
)

RETRIEVAL_ERROR = (
    "I couldn't reach the search index just now, so I can't answer reliably. Please try again in a moment."
)

LLM_ERROR = (
    "The language model service is busy or unavailable right now. Please try again in a moment."
)
