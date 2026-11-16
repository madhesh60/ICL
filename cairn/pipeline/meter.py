"""Counting what a run costs, per kind of call.

"Which stage eats the budget?" is a question you will ask in every project.
The meter wraps a backend and adds up calls, tokens and seconds for each tag
("score", "think_vote", "induce_program", ...).  It also reports how much of
the prompt the server took from its cache, which is the number that tells you
whether the cache-aligned prompt layout is paying off.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass

from cairn.llm.base import GenRequest, GenResponse, LLMBackend


@dataclass
class TagStats:
    calls: int = 0
    cache_file_hits: int = 0  # answered from our own disk cache (no server work at all)
    prompt_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens the server reused from its KV cache
    completion_tokens: int = 0
    seconds: float = 0.0


class UsageMeter:
    def __init__(self, inner: LLMBackend) -> None:
        self.inner = inner
        self.model = inner.model
        self._stats: dict[str, TagStats] = defaultdict(TagStats)
        self._lock = threading.Lock()

    def generate(self, request: GenRequest) -> GenResponse:
        response = self.inner.generate(request)
        with self._lock:
            stats = self._stats[request.tag or "untagged"]
            stats.calls += 1
            if response.from_cache:
                stats.cache_file_hits += 1
            else:
                stats.prompt_tokens += response.usage.prompt_tokens
                stats.cached_tokens += response.usage.cached_tokens
                stats.completion_tokens += response.usage.completion_tokens
                stats.seconds += response.latency_s
        return response

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for tag, s in sorted(self._stats.items()):
                out[tag] = {
                    "calls": s.calls,
                    "cache_file_hits": s.cache_file_hits,
                    "prompt_tokens": s.prompt_tokens,
                    "cached_tokens": s.cached_tokens,
                    "kv_cache_share": round(s.cached_tokens / s.prompt_tokens, 4)
                    if s.prompt_tokens
                    else 0.0,
                    "completion_tokens": s.completion_tokens,
                    "seconds": round(s.seconds, 2),
                }
            return out

    def total_calls(self) -> int:
        with self._lock:
            return sum(s.calls for s in self._stats.values())
