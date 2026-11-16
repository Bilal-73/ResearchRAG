"""Output parsers shared by every JSON-returning chain."""
from __future__ import annotations

import json
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(raw: str) -> dict | None:
    """Tolerant JSON-object parser: strips code fences and falls back to the
    first {...} span if the model wrapped the JSON in prose."""
    text = _FENCE_RE.sub("", raw.strip())
    for candidate in (text, _first_brace_span(text)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _first_brace_span(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else None


class JsonObjectOutputParser(BaseOutputParser[dict]):
    """LangChain parser around `parse_json_object`. Raises OutputParserException on
    junk, so a chain's `.with_fallbacks()` can substitute a safe default."""

    @property
    def _type(self) -> str:
        return "json_object"

    def parse(self, text: str) -> dict:
        value = parse_json_object(text)
        if value is None:
            raise OutputParserException(f"Expected a JSON object, got: {text[:120]!r}", llm_output=text)
        return value
