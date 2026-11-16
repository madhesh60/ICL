import threading

import pytest

from cairn.decoding.constraints import Choice
from cairn.llm import (
    BadRequestError,
    CachedBackend,
    DemoReaderBackend,
    FunctionBackend,
    GenRequest,
    OpenAICompatBackend,
    RetryPolicy,
    RouterBackend,
    ScriptedBackend,
    ServerUnavailableError,
)
from tests.fake_server import FakeLLMServer

FAST = RetryPolicy(
    max_attempts=3, base_delay_s=0.0, max_delay_s=0.0, breaker_threshold=3, cooldown_s=60
)


def _req(**kwargs) -> GenRequest:
    return GenRequest(messages=(("user", "hello"),), **kwargs)


def test_payload_carries_thinking_switch_constraint_and_logprobs():
    backend = OpenAICompatBackend("http://x/v1", "Qwen3-4B")
    body = backend.payload(
        _req(constraint=Choice(("Y", "N")), top_logprobs=5, thinking=False, seed=3)
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["structured_outputs"] == {"choice": ["Y", "N"]}
    assert body["logprobs"] is True and body["top_logprobs"] == 5 and body["seed"] == 3
    assert "tag" not in body


def test_success_parses_text_reasoning_logprobs_and_cached_tokens():
    with FakeLLMServer() as server:
        backend = OpenAICompatBackend(server.base_url, retry=FAST)
        first = backend.generate(_req(thinking=True, top_logprobs=3))
        assert first.texts == ["OK"] and first.reasoning == ["reasoning here"]
        assert first.first_token_logprobs[0]["Sad"] == pytest.approx(-0.4)
        second = backend.generate(_req())
        assert second.usage.cached_tokens > 0  # same prompt beginning as before


def test_inline_think_block_is_split_from_the_answer():
    with FakeLLMServer() as server:
        server.include_think_inline = True
        response = OpenAICompatBackend(server.base_url, retry=FAST).generate(_req(thinking=True))
        assert response.texts == ["OK"] and response.reasoning == ["reasoning here"]


def test_retries_transient_errors_then_succeeds():
    with FakeLLMServer() as server:
        server.fail_statuses = [503, 429]
        response = OpenAICompatBackend(server.base_url, retry=FAST).generate(_req())
        assert response.texts == ["OK"] and len(server.requests) == 3


def test_bad_request_is_not_retried():
    with FakeLLMServer() as server:
        server.fail_statuses = [400]
        with pytest.raises(BadRequestError):
            OpenAICompatBackend(server.base_url, retry=FAST).generate(_req())
        assert len(server.requests) == 1


def test_gives_up_and_circuit_breaker_fails_fast():
    with FakeLLMServer() as server:
        server.fail_statuses = [500] * 10
        backend = OpenAICompatBackend(server.base_url, retry=FAST)
        with pytest.raises(ServerUnavailableError):
            backend.generate(_req())
        sent = len(server.requests)
        with pytest.raises(ServerUnavailableError, match="circuit breaker"):
            backend.generate(_req())
        assert len(server.requests) == sent  # the open breaker did not even call the server


def test_unreachable_server_raises_server_unavailable():
    backend = OpenAICompatBackend("http://127.0.0.1:9/v1", retry=FAST, timeout_s=1)
    with pytest.raises(ServerUnavailableError):
        backend.generate(_req())


def test_disk_cache_reuses_deterministic_answers_only(tmp_path):
    inner = FunctionBackend(lambda r: f"answer {r.temperature}")
    cached = CachedBackend(inner, tmp_path / "c.sqlite")
    assert cached.generate(_req()).texts == ["answer 0.0"]
    again = cached.generate(_req())
    assert again.from_cache and inner.calls == 1
    cached.generate(_req(temperature=0.7))
    cached.generate(_req(temperature=0.7))
    assert inner.calls == 3  # random requests are never frozen
    cached.generate(_req(temperature=0.7, seed=1))
    cached.generate(_req(temperature=0.7, seed=1))
    assert inner.calls == 4  # ...unless a seed makes them repeatable
    cached.close()
    reopened = CachedBackend(inner, tmp_path / "c.sqlite")
    assert reopened.generate(_req()).from_cache  # survives a restart


def test_cache_key_ignores_the_tag_but_not_the_prompt():
    assert _req(tag="a").cache_key("m") == _req(tag="b").cache_key("m")
    assert _req().cache_key("m") != GenRequest(messages=(("user", "bye"),)).cache_key("m")
    assert _req().cache_key("m1") != _req().cache_key("m2")


def test_cache_is_safe_under_threads(tmp_path):
    cached = CachedBackend(FunctionBackend(lambda r: r.messages[0][1]), tmp_path / "t.sqlite")
    errors = []

    def work(i):
        try:
            assert cached.generate(GenRequest(messages=(("user", str(i % 5)),))).texts == [
                str(i % 5)
            ]
        except Exception as err:  # pragma: no cover - only on failure
            errors.append(err)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_scripted_and_router_backends():
    scripted = ScriptedBackend(["one", ["a", "b"]])
    router = RouterBackend(
        [(lambda r: r.tag == "x", FunctionBackend(lambda r: "routed"))], scripted
    )
    assert router.generate(_req(tag="x")).texts == ["routed"]
    assert router.generate(_req()).texts == ["one"]
    assert router.generate(_req(n=2)).texts == ["a", "b"]
    with pytest.raises(AssertionError):
        scripted.generate(_req())


def test_demo_reader_answers_like_the_nearest_example():
    reader = DemoReaderBackend()
    content = "Input: the cat purrs\nOutput: animal\n\nInput: stocks fell\nOutput: finance\n\nInput: the cat sleeps"
    request = GenRequest(
        messages=(("user", content),), top_logprobs=2, constraint=Choice(("animal", "finance"))
    )
    response = reader.generate(request)
    assert response.texts == ["animal"]
    assert response.first_token_logprobs[0]["animal"] > response.first_token_logprobs[0]["finance"]
