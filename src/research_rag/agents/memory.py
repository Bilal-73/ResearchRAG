"""Conversation memory management: bounded, summarised, and PII-free.

* `history` keeps the last N turns verbatim; once it grows past that, the
  oldest turns are folded into `summary` by one small LLM call, so context
  size stays flat however long the chat runs.
* Only redacted user text is ever stored (the guard runs first).
* The classifier and rewriter get a compact view (`format_recent_turns`) with
  long answers truncated — a follow-up needs the *topic*, not 2000 characters
  of the previous answer.
"""
from __future__ import annotations

import logging

from research_rag.llm import LLMClient
from .prompts import SUMMARY_SYSTEM

logger = logging.getLogger(__name__)

MAX_STORED_ANSWER_CHARS = 1800


def format_recent_turns(history: list[dict], n_turns: int = 2, answer_chars: int = 350) -> str:
    """Last `n_turns` exchanges as short 'User:/Assistant:' lines."""
    lines = []
    for msg in history[-(n_turns * 2):]:
        who = "User" if msg["role"] == "user" else "Assistant"
        text = msg["content"]
        if msg["role"] == "assistant" and len(text) > answer_chars:
            text = text[:answer_chars].rstrip() + " …"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def append_turn(history: list[dict], user_text: str, answer: str, intent: str) -> list[dict]:
    return history + [
        {"role": "user", "content": user_text, "intent": intent},
        {"role": "assistant", "content": answer[:MAX_STORED_ANSWER_CHARS], "intent": intent},
    ]


def fold_old_turns(history: list[dict], summary: str, max_turns: int, llm: LLMClient | None) -> tuple[list[dict], str]:
    """If more than `max_turns` exchanges are stored, summarise the oldest and drop them."""
    max_messages = max_turns * 2
    if len(history) <= max_messages:
        return history, summary

    overflow = history[:-max_messages]
    kept = history[-max_messages:]
    logger.info(f"Memory: folding {len(overflow) // 2} old turn(s) into the rolling summary")

    if llm is None:
        return kept, summary
    transcript = format_recent_turns(overflow, n_turns=len(overflow), answer_chars=300)
    try:
        new_summary = llm.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": f"EXISTING SUMMARY:\n{summary or '(none)'}\n\nTURNS TO FOLD IN:\n{transcript}"},
            ],
            temperature=0.0, max_tokens=220, purpose="summarize",
        ).strip()
    except Exception as error:  # noqa: BLE001 — losing a summary is acceptable; losing the chat is not
        logger.warning(f"Summarisation failed ({error}); dropping old turns without a summary")
        return kept, summary
    return kept, new_summary
