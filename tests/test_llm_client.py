from types import SimpleNamespace

import pytest

from research_rag.llm import LLMClient, LLMError, parse_json_object


class HTTPError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"http {status}")
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after": str(retry_after)} if retry_after else {})


def _client(outcomes, sleeps):
    it = iter(outcomes)

    def create(**kw):
        o = next(it)
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=o))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return LLMClient(client=fake, deployment="d", max_retries=3, sleep=sleeps.append)


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
