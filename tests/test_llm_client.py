from types import SimpleNamespace

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from research_rag.llm import LLMClient, LLMError, parse_json_object


class HTTPError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"http {status}")
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after": str(retry_after)} if retry_after else {})


class SequenceChatModel(BaseChatModel):
    """Returns (or raises) the scripted outcomes in order."""
    outcomes: list
    seen_kwargs: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-sequence"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_kwargs.append(kwargs)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=o))])


def _client(outcomes, sleeps):
    return LLMClient(SequenceChatModel(outcomes=list(outcomes)), max_retries=3, sleep=sleeps.append)


def test_retries_429_honouring_retry_after_then_succeeds():
    sleeps = []
    c = _client([HTTPError(429, retry_after=7), HTTPError(503), "ok"], sleeps)
    assert c.chat([{"role": "user", "content": "x"}]) == "ok"
    assert sleeps[0] == 7 and len(sleeps) == 2
    assert c.stats["retries"] == 2 and c.stats["rate_limit_hits"] == 1


def test_gives_up_after_max_retries():
    c = _client([HTTPError(500)] * 5, [])
    with pytest.raises(LLMError):
        c.chat([{"role": "user", "content": "x"}])
    assert c.stats["failures"] == 1


def test_non_retryable_error_fails_immediately():
    sleeps = []
    c = _client([HTTPError(400)], sleeps)
    with pytest.raises(LLMError):
        c.chat([{"role": "user", "content": "x"}])
    assert sleeps == []


def test_chat_json_returns_default_on_garbage_and_on_failure():
    assert _client(["not json at all"], []).chat_json([{"role": "user", "content": "x"}], purpose="t", default={"a": 1}) == {"a": 1}
    assert _client([HTTPError(400)], []).chat_json([{"role": "user", "content": "x"}], purpose="t", default={"b": 2}) == {"b": 2}


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert parse_json_object("[1,2]") is None


def test_json_mode_and_limits_are_bound_onto_the_chat_model():
    model = SequenceChatModel(outcomes=['{"a": 1}'])
    LLMClient(model).chat_json([{"role": "user", "content": "x"}], purpose="t", max_tokens=42)
    assert model.seen_kwargs[0]["response_format"] == {"type": "json_object"}
    assert model.seen_kwargs[0]["max_tokens"] == 42 and model.seen_kwargs[0]["temperature"] == 0.0


def test_json_chain_parses_and_falls_back_to_default():
    prompt = ChatPromptTemplate.from_messages([("system", "sys"), ("human", "{q}")])
    ok = LLMClient(SequenceChatModel(outcomes=['```json\n{"a": 1}\n```'])).json_chain(prompt, purpose="t")
    assert ok.invoke({"q": "x"}) == {"a": 1}
    junk = LLMClient(SequenceChatModel(outcomes=["no json here"])).json_chain(prompt, purpose="t", default={"d": 0})
    assert junk.invoke({"q": "x"}) == {"d": 0}
    down = LLMClient(SequenceChatModel(outcomes=[HTTPError(400)])).json_chain(prompt, purpose="t", default={"d": 1})
    assert down.invoke({"q": "x"}) == {"d": 1}


def test_text_chain_retries_through_the_runnable():
    sleeps = []
    prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
    chain = _client([HTTPError(503), "fine"], sleeps).text_chain(prompt, purpose="t")
    assert chain.invoke({"q": "x"}) == "fine" and len(sleeps) == 1
