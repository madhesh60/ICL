"""Client for any server that speaks the OpenAI chat API (vLLM, FlagScale, llama.cpp ...).

Networks fail.  A run that labels 4,000 inputs will see timeouts, dropped
connections and "server busy" replies.  This client turns those into three
clear outcomes:

    it worked                      -> GenResponse
    our request is wrong           -> BadRequestError       (never retried)
    the server is in trouble       -> ServerUnavailableError (after retries)

Two classic patterns do the work:

* Retry with exponential backoff and jitter: wait 1s, 2s, 4s ... plus a random
  bit, so that many waiting clients do not all hit the server at the same instant.
* Circuit breaker: after several failures in a row, stop calling for a while and
  fail fast.  Hammering a server that is down only keeps it down, and the run
  journal lets us continue later exactly where we stopped.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from cairn.decoding.constraints import to_request_fields
from cairn.decoding.parse import split_reasoning
from cairn.llm.base import BadRequestError, GenRequest, GenResponse, ServerUnavailableError, Usage

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    # Circuit breaker: after this many failed calls in a row, refuse new calls for `cooldown_s`.
    breaker_threshold: int = 8
    cooldown_s: float = 60.0

    def delay(self, attempt: int, rng: random.Random) -> float:
        """Full jitter: a random wait between 0 and the exponential cap."""
        return rng.uniform(0.0, min(self.max_delay_s, self.base_delay_s * (2**attempt)))


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if time.monotonic() - self._opened_at >= self._cooldown_s:
                # Cool-down is over: let one call through to test the water.
                self._opened_at = None
                self._failures = self._threshold - 1
                return
            raise ServerUnavailableError("circuit breaker is open: the server keeps failing")

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._failures = 0
                return
            self._failures += 1
            if self._failures >= self._threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                log.error("circuit breaker opened after %d failures in a row", self._failures)


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:2026/v1",
        model: str = "Qwen3-4B",
        *,
        api_key: str = "EMPTY",
        timeout_s: float = 600.0,
        retry: RetryPolicy | None = None,
        structured_api: str = "structured_outputs",
        thinking_control: bool = True,
        seed: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retry = retry or RetryPolicy()
        self.structured_api = structured_api
        self.thinking_control = thinking_control
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._breaker = CircuitBreaker(self.retry.breaker_threshold, self.retry.cooldown_s)
        self._rng = random.Random(seed)
        self._local = threading.local()

    # One HTTP session per thread: sessions keep connections open (fast) but are
    # not safe to share between threads.
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def payload(self, request: GenRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": request.message_dicts(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
        }
        if request.top_k is not None:
            body["top_k"] = request.top_k
        if request.seed is not None:
            body["seed"] = request.seed
        if request.stop:
            body["stop"] = list(request.stop)
        if request.top_logprobs > 0:
            body["logprobs"] = True
            body["top_logprobs"] = request.top_logprobs
        if self.thinking_control:
            # Qwen3 thinks by default; this switch is read by its chat template.
            body["chat_template_kwargs"] = {"enable_thinking": request.thinking}
        body.update(to_request_fields(request.constraint, self.structured_api))
        return body

    def generate(self, request: GenRequest) -> GenResponse:
        body = self.payload(request)
        url = f"{self.base_url}/chat/completions"
        last_error = "no attempt made"
        for attempt in range(self.retry.max_attempts):
            self._breaker.before_call()
            started = time.monotonic()
            try:
                reply = self._session().post(url, json=body, headers=self._headers, timeout=self.timeout_s)
            except requests.RequestException as err:
                last_error = f"{type(err).__name__}: {err}"
                self._breaker.record(ok=False)
            else:
                if reply.status_code == 200:
                    try:
                        parsed = self._parse(reply.json(), time.monotonic() - started)
                    except (ValueError, KeyError, IndexError, TypeError) as err:
                        last_error = f"malformed reply: {err!r}"
                        self._breaker.record(ok=False)
                    else:
                        self._breaker.record(ok=True)
                        return parsed
                elif reply.status_code in _RETRYABLE_STATUS:
                    last_error = f"HTTP {reply.status_code}: {reply.text[:200]}"
                    self._breaker.record(ok=False)
                else:
                    # 400/401/404/422...: the request itself is the problem.  The server
                    # is healthy, so this does not count against the circuit breaker.
                    self._breaker.record(ok=True)
                    raise BadRequestError(f"HTTP {reply.status_code}: {reply.text[:500]}")
            if attempt + 1 < self.retry.max_attempts:
                wait = self.retry.delay(attempt, self._rng)
                log.warning("call failed (%s); retry %d in %.1fs", last_error, attempt + 1, wait)
                time.sleep(wait)
        raise ServerUnavailableError(
            f"gave up after {self.retry.max_attempts} attempts; last error: {last_error}"
        )

    def _parse(self, data: dict[str, Any], latency_s: float) -> GenResponse:
        texts: list[str] = []
        reasoning: list[str] = []
        logprobs: list[dict[str, float]] = []
        finish: list[str] = []
        for choice in data["choices"]:
            message = choice["message"]
            content = message.get("content") or ""
            # Newer vLLM calls the field "reasoning", older "reasoning_content".  With
            # no reasoning parser on the server the <think> block is still in the text.
            thought = message.get("reasoning") or message.get("reasoning_content") or ""
            if not thought:
                thought, content = split_reasoning(content)
            texts.append(content.strip())
            reasoning.append(thought)
            finish.append(choice.get("finish_reason") or "")
            first: dict[str, float] = {}
            tokens = (choice.get("logprobs") or {}).get("content") or []
            if tokens:
                for entry in tokens[0].get("top_logprobs") or []:
                    first[entry["token"]] = float(entry["logprob"])
            logprobs.append(first)
        raw = data.get("usage") or {}
        details = raw.get("prompt_tokens_details") or {}
        usage = Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or 0),
        )
        return GenResponse(
            texts=texts,
            reasoning=reasoning,
            first_token_logprobs=logprobs,
            finish_reasons=finish,
            usage=usage,
            latency_s=latency_s,
        )

    def list_models(self) -> list[str]:
        reply = self._session().get(f"{self.base_url}/models", headers=self._headers, timeout=30)
        reply.raise_for_status()
        return [m["id"] for m in reply.json().get("data", [])]
