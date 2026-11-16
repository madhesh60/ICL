"""Measuring a prompt design without calling a model.

Three questions about any way of building the long prompt can be answered with
plain arithmetic on real data, long before a GPU is involved:

    coverage     How much of what an input contains (its informative words and
                 word pairs) appears somewhere among the chosen examples?
                 Higher = the examples are a better "textbook" for this input.

    streaks      How long is the longest run of one label in a row?  Long runs
                 push a model towards that label.

    cache share  If the server remembers prompt beginnings, what share of all
                 prompt tokens does it NOT have to read again?  This is most of
                 the compute bill for 30K-token prompts.

These are proxies, not accuracy.  They tell us whether the prompt is built the
way we intended; only a real model run tells us how much accuracy that buys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from cairn.context.builder import ContextBuilder
from cairn.tokens import TokenCounter, common_prefix_len
from cairn.types import Query


@dataclass(frozen=True)
class ContextReport:
    n_queries: int
    mean_bulk_examples: float
    mean_demo_tokens: float
    min_demo_tokens: int
    all_meet_minimum: bool
    coverage: float  # IDF-weighted share of query features found in the big block
    longest_label_streak: int
    label_entropy_bits: float

    def to_json(self) -> dict[str, float | int | bool]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class CacheReport:
    n_prompts: int
    total_tokens: int
    cached_tokens: int
    cache_share: float
    tokens_to_read: int  # what the server must actually process

    def to_json(self) -> dict[str, float | int]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _streak_and_entropy(labels: Sequence[str]) -> tuple[int, float]:
    longest = current = 0
    previous: str | None = None
    counts: dict[str, int] = {}
    for label in labels:
        current = current + 1 if label == previous else 1
        longest = max(longest, current)
        previous = label
        counts[label] = counts.get(label, 0) + 1
    probs = np.array(list(counts.values()), dtype=np.float64) / max(len(labels), 1)
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
    return longest, entropy


def context_report(builder: ContextBuilder, queries: Sequence[Query]) -> ContextReport:
    """Coverage, label streaks and the token-minimum check over a set of inputs."""
    bulk_features: dict[int, set[str]] = {}
    for cluster in range(builder.n_clusters):
        chosen = builder._global + builder._cluster_blocks[cluster]
        bulk_features[cluster] = {f for i in chosen for f in builder._feats[i]}
    coverages: list[float] = []
    demo_tokens: list[int] = []
    bulk_sizes: list[int] = []
    for query in queries:
        prompt = builder.build(query)
        demo_tokens.append(prompt.stats.unique_demo_tokens)
        bulk_sizes.append(prompt.stats.n_bulk)
        feats = set(builder._features(builder.spec.view(query.input)))
        weight = {f: builder._bm25.idf(f) for f in feats}
        total = sum(weight.values())
        if total > 0:
            found = sum(w for f, w in weight.items() if f in bulk_features[prompt.cluster])
            coverages.append(found / total)
    labels = [e.output for e in builder.bulk_examples(0)]
    streak, entropy = (
        _streak_and_entropy(labels) if builder.spec.kind == "classification" else (0, 0.0)
    )
    return ContextReport(
        n_queries=len(queries),
        mean_bulk_examples=float(np.mean(bulk_sizes)) if bulk_sizes else 0.0,
        mean_demo_tokens=float(np.mean(demo_tokens)) if demo_tokens else 0.0,
        min_demo_tokens=min(demo_tokens) if demo_tokens else 0,
        all_meet_minimum=all(t >= builder.min_required for t in demo_tokens),
        coverage=float(np.mean(coverages)) if coverages else 0.0,
        longest_label_streak=streak,
        label_entropy_bits=entropy,
    )


def render_chat(messages: Sequence[dict[str, str]]) -> str:
    """The text a Qwen-style chat template produces (ChatML), close enough for counting."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def simulate_prefix_cache(
    builder: ContextBuilder,
    queries: Sequence[Query],
    counter: TokenCounter,
    *,
    grouped: bool = True,
    slots: int = 4,
    block: int = 16,
) -> CacheReport:
    """Replay the prompts through a simple model of a server-side prefix cache.

    The model: the server remembers the `slots` most recently used prompt
    beginnings (one per cluster here) and can reuse the longest common prefix
    with any of them, in whole blocks of `block` tokens.  That mirrors how vLLM's
    automatic prefix caching behaves (hash-addressed 16-token blocks, least
    recently used blocks evicted when memory runs out).

    grouped=True   send prompts cluster by cluster (what Cairn's scheduler does)
    grouped=False  send them in arrival order
    """
    order = list(range(len(queries)))
    if grouped:
        order.sort(key=lambda i: builder.route(queries[i].input))
    remembered: dict[int, np.ndarray] = {}  # cluster -> tokens of its latest prompt
    recency: list[int] = []
    total = cached = 0
    for i in order:
        prompt = builder.build(queries[i])
        tokens = np.asarray(counter.encode(render_chat(prompt.messages)), dtype=np.int64)
        reuse = max((common_prefix_len(tokens, old) for old in remembered.values()), default=0)
        reuse -= reuse % block
        total += len(tokens)
        cached += reuse
        remembered[prompt.cluster] = tokens
        if prompt.cluster in recency:
            recency.remove(prompt.cluster)
        recency.append(prompt.cluster)
        while len(recency) > slots:
            remembered.pop(recency.pop(0), None)
    return CacheReport(
        n_prompts=len(order),
        total_tokens=total,
        cached_tokens=cached,
        cache_share=cached / total if total else 0.0,
        tokens_to_read=total - cached,
    )
