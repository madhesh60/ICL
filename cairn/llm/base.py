"""The contract between the pipeline and any language-model server.

The pipeline only ever builds a `GenRequest` and reads a `GenResponse`.  It does
not know whether the answer came from a real vLLM server, a saved cache file or
a fake model used in tests.  That one seam is what makes the whole system
testable on a laptop without a GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from cairn.decoding.constraints import Choice, Constraint, Regex


class LLMError(RuntimeError):
    """Base class for everything that can go wrong while calling the model."""


class BadRequestError(LLMError):
    """The server understood us and said no (for example: prompt too long).

    Retrying the same request cannot help, so this is never retried.
    """


class ServerUnavailableError(LLMError):
    """The server did not answer properly even after retries, or the circuit is open."""


@dataclass(frozen=True)
class GenRequest:
    messages: tuple[tuple[str, str], ...]  # (role, content) pairs
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    n: int = 1
    seed: int | None = None
    stop: tuple[str, ...] = ()
    thinking: bool = False
    constraint: Constraint | None = None
    # Ask for the k most likely first tokens with their log-probabilities.
    top_logprobs: int = 0
    # Free text for logs and metrics ("score", "think", "induce", ...).  Not sent to the server.
    tag: str = ""

    @classmethod
    def from_messages(cls, messages: list[dict[str, str]], **kwargs: Any) -> GenRequest:
        return cls(messages=tuple((m["role"], m["content"]) for m in messages), **kwargs)

    def message_dicts(self) -> list[dict[str, str]]:
        return [{"role": role, "content": content} for role, content in self.messages]

    def cache_key(self, model: str) -> str:
        """A fingerprint of everything that can change the answer."""
        payload = asdict(self)
        payload.pop("tag")
        if isinstance(self.constraint, Choice):
            payload["constraint"] = ["choice", list(self.constraint.options)]
        elif isinstance(self.constraint, Regex):
            payload["constraint"] = ["regex", self.constraint.pattern]
        payload["model"] = model
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def deterministic(self) -> bool:
        """True when asking twice must give the same answer (safe to cache)."""
        return self.temperature == 0.0 or self.seed is not None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens the server did not have to re-read

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens


@dataclass
class GenResponse:
    texts: list[str]
    reasoning: list[str] = field(default_factory=list)
    # For each returned text: {token string: log-probability} for the first generated token.
    first_token_logprobs: list[dict[str, float]] = field(default_factory=list)
    finish_reasons: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    from_cache: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "texts": self.texts,
            "reasoning": self.reasoning,
            "first_token_logprobs": self.first_token_logprobs,
            "finish_reasons": self.finish_reasons,
            "usage": asdict(self.usage),
            "latency_s": self.latency_s,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> GenResponse:
        return cls(
            texts=list(row["texts"]),
            reasoning=list(row.get("reasoning", [])),
            first_token_logprobs=[dict(d) for d in row.get("first_token_logprobs", [])],
            finish_reasons=list(row.get("finish_reasons", [])),
            usage=Usage(**row.get("usage", {})),
            latency_s=float(row.get("latency_s", 0.0)),
        )


@runtime_checkable
class LLMBackend(Protocol):
    model: str

    def generate(self, request: GenRequest) -> GenResponse: ...
