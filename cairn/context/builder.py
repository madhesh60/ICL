"""The context builder: turns "5,000 labelled examples + one input" into a prompt.

The prompt is laid out like a tree, so that prompts share as much of their
beginning as possible:

    [ system: task + output rule ]                         same for the whole task
    [ shared block:  examples every prompt gets ]          same for the whole task
    [ cluster block: examples for this kind of input ]     same inside a cluster
    [ session block: answers given earlier in the run ]    only ever grows at the end
    [ tail:          the nearest examples to THIS input ]  different every time
    [ input to label ]

Why this order?

* A language model reads left to right and keeps notes (the "KV cache") for
  everything it has read.  If two prompts start with the same text, the server
  can reuse the notes and only read the part that differs.  So everything that
  repeats goes first, everything that changes goes last.
* A small model remembers the end of a long prompt best.  So the examples that
  matter most for this input (the nearest ones) sit right before the input.

The shared and cluster blocks together always hold at least the required
minimum of example tokens, so the "30K tokens of examples" rule holds for every
single prompt no matter what the tail looks like.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cairn.context import render
from cairn.retrieval import (
    BM25Index,
    HashedTfidf,
    lexical_features,
    mmr_select,
    spherical_kmeans,
    word_tokens,
)
from cairn.selection import budgeted_coverage
from cairn.tasks.base import TaskSpec, View
from cairn.tokens import TokenCounter
from cairn.types import Example, Query

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextConfig:
    # Minimum tokens of examples in every prompt; None = the task's official minimum.
    min_demo_tokens: int | None = None
    # How the big block of examples is chosen.
    selection: Literal["coverage", "random", "first"] = "coverage"
    # "pool": cover what the labelled pool looks like.
    # "queries": cover what the inputs we must label look like (when known up front).
    coverage_target: Literal["pool", "queries"] = "queries"
    # 0 keeps each group's natural share, 1 gives every group an equal share.
    label_balance: float = 0.5
    bulk_order: Literal["interleave", "random", "selection"] = "interleave"
    # Number of cluster-specific blocks (1 = one prompt beginning for everything).
    n_clusters: int = 1
    # Share of the example budget that is common to all clusters.
    global_frac: float = 0.5
    # The per-input tail.
    tail_k: int = 12
    tail_per_label: int = 2
    tail_mmr: float = 1.0  # 1.0 = pure relevance, lower = more variety
    tail_as_turns: bool = True
    restate_rule: bool = True
    seed: int = 0


@dataclass(frozen=True)
class ContextStats:
    bulk_tokens: int
    session_tokens: int
    tail_tokens: int
    unique_demo_tokens: int
    n_bulk: int
    n_session: int
    n_tail: int
    prompt_tokens: int
    min_required: int

    @property
    def meets_minimum(self) -> bool:
        return self.unique_demo_tokens >= self.min_required


@dataclass(frozen=True)
class BuiltPrompt:
    messages: list[dict[str, str]]
    stats: ContextStats
    cluster: int
    # Identifies the reusable beginning of the prompt.  Prompts with the same key
    # share their cached part; the scheduler groups by it.
    prefix_key: str
    tail_ids: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


class ContextBuilder:
    def __init__(
        self,
        spec: TaskSpec,
        pool: Sequence[Example],
        counter: TokenCounter,
        config: ContextConfig | None = None,
        *,
        definition: str = "",
        guidelines: str = "",
        target_inputs: Sequence[str] | None = None,
    ) -> None:
        if not pool:
            raise ValueError("the example pool is empty")
        self.spec = spec
        self.pool = list(pool)
        self.counter = counter
        self.config = config or ContextConfig()
        self.definition = definition
        self.guidelines = guidelines
        self._rng = np.random.default_rng(self.config.seed)

        required = self.config.min_demo_tokens or spec.min_context_tokens
        # An inexact token counter gets a safety margin on top of the minimum.
        self.min_required = required
        self._budget = int(np.ceil(required * (1.0 + counter.margin)))

        self._views: list[View] = [spec.view(e.input) for e in self.pool]
        self._feats: list[list[str]] = [self._features(v) for v in self._views]
        self._bm25 = BM25Index(self._feats)
        self._rendered: list[str] = [render.render_demo(spec, e) for e in self.pool]
        self._costs: list[int] = [counter.count(r) for r in self._rendered]
        self._strata: dict[str, np.ndarray] = {}
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for i, view in enumerate(self._views):
            by_stratum[view.stratum].append(i)
        for stratum, members in by_stratum.items():
            mask = np.zeros(len(self.pool), dtype=bool)
            mask[members] = True
            self._strata[stratum] = mask
        self._labels: list[str] = [e.output for e in self.pool]

        total_pool_tokens = sum(self._costs)
        if total_pool_tokens < self._budget:
            raise ValueError(
                f"the pool holds {total_pool_tokens} tokens of examples but {self._budget} are "
                "required; lower the reserved split sizes or min_demo_tokens"
            )

        # Feature ids for the coverage objective.
        self._term_id: dict[str, int] = {}
        self._doc_feats: list[tuple[np.ndarray, np.ndarray]] = []
        for feats in self._feats:
            ids = sorted({self._term_id.setdefault(t, len(self._term_id)) for t in feats})
            arr = np.asarray(ids, dtype=np.int64)
            self._doc_feats.append((arr, np.ones(len(arr), dtype=np.float64)))
        self._idf = np.zeros(len(self._term_id), dtype=np.float64)
        for term, tid in self._term_id.items():
            self._idf[tid] = self._bm25.idf(term)

        target_feats = (
            [self._features(spec.view(text)) for text in target_inputs]
            if target_inputs is not None and self.config.coverage_target == "queries"
            else self._feats
        )

        # Clusters (only when asked for): fitted on the pool, so routing also works
        # for inputs that arrive later, one at a time.
        self._vectorizer: HashedTfidf | None = None
        self._vectors: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        n_clusters = max(1, self.config.n_clusters)
        if n_clusters > 1 or self.config.tail_mmr < 1.0:
            self._vectorizer = HashedTfidf().fit(self._feats)
            self._vectors = self._vectorizer.transform(self._feats)
        if n_clusters > 1:
            assert self._vectors is not None
            self._centroids, _ = spherical_kmeans(self._vectors, n_clusters, seed=self.config.seed)
            n_clusters = len(self._centroids)
        self.n_clusters = n_clusters

        target_cluster = (
            self._route_features(target_feats)
            if n_clusters > 1
            else np.zeros(len(target_feats), int)
        )
        global_frac = 1.0 if n_clusters == 1 else float(np.clip(self.config.global_frac, 0.0, 1.0))
        global_budget = round(self._budget * global_frac)
        self._global: list[int] = (
            self._select(global_budget, self._weights(target_feats), already=())
            if global_budget
            else []
        )
        global_tokens = sum(self._costs[i] for i in self._global)
        self._cluster_blocks: list[list[int]] = []
        for c in range(n_clusters):
            remaining = self._budget - global_tokens
            if remaining <= 0:
                self._cluster_blocks.append([])
                continue
            members = [f for f, cl in zip(target_feats, target_cluster, strict=True) if cl == c]
            weights = self._weights(members or target_feats)
            self._cluster_blocks.append(self._select(remaining, weights, already=self._global))

        self._system = render.system_prompt(spec, definition, guidelines)
        self._bulk_text: list[str] = []
        self._bulk_tokens: list[int] = []
        self._bulk_ids: list[set[str]] = []
        for c in range(n_clusters):
            chosen = self._global + self._cluster_blocks[c]
            self._bulk_text.append(render.BULK_HEADING + "".join(self._rendered[i] for i in chosen))
            self._bulk_tokens.append(sum(self._costs[i] for i in chosen))
            self._bulk_ids.append({self.pool[i].id for i in chosen})
        self._overhead = counter.count(self._system) + counter.count(render.BULK_HEADING) + 24

    # ------------------------------------------------------------------ features

    def _features(self, view: View) -> list[str]:
        feats = lexical_features(view.text, bigrams=True, chars=self.spec.char_features)
        if view.boosted:
            boosted = word_tokens(view.boosted)
            feats.extend(f"w:{w}" for w in boosted)
            feats.extend(f"k:{w}" for w in boosted)
        return feats

    def _weights(self, target_feats: Sequence[Sequence[str]]) -> np.ndarray:
        """Weight of a feature = how rare it is x how often our targets contain it.

        Rare features tell examples apart (worth covering); features that the
        targets never contain are not worth spending tokens on.
        """
        seen = np.zeros(len(self._term_id), dtype=np.float64)
        for feats in target_feats:
            for tid in {self._term_id[t] for t in feats if t in self._term_id}:
                seen[tid] += 1.0
        return self._idf * seen / max(len(target_feats), 1)

    def _route_features(self, feats: Sequence[Sequence[str]]) -> np.ndarray:
        assert self._vectorizer is not None and self._centroids is not None
        if not feats:
            return np.zeros(0, dtype=int)
        return np.argmax(self._vectorizer.transform(feats) @ self._centroids.T, axis=1)

    # ----------------------------------------------------------------- selection

    def _group(self, i: int) -> str:
        stratum = self._views[i].stratum
        if self.spec.kind == "classification":
            return f"{stratum}|{self._labels[i]}"
        return stratum

    def _select(self, budget: int, weights: np.ndarray, already: Sequence[int]) -> list[int]:
        cfg = self.config
        taken = set(already)
        candidates = [i for i in range(len(self.pool)) if i not in taken]
        if cfg.selection == "first":
            chosen = self._fill(candidates, budget)
        elif cfg.selection == "random":
            order = np.random.default_rng(cfg.seed).permutation(len(candidates))
            chosen = self._fill([candidates[j] for j in order], budget)
        else:
            groups = [self._group(i) for i in range(len(self.pool))]
            counts = Counter(groups[i] for i in candidates)
            n = sum(counts.values())
            beta = float(np.clip(cfg.label_balance, 0.0, 1.0))
            share = {g: (1 - beta) * c / n + beta / len(counts) for g, c in counts.items()}
            result = budgeted_coverage(
                self._doc_feats,
                weights,
                self._costs,
                budget,
                groups=groups,
                group_share=share,
                candidates=candidates,
                already_chosen=tuple(already),
            )
            chosen = result.order
        return self._order_bulk(chosen)

    def _fill(self, ordered: Sequence[int], budget: int) -> list[int]:
        chosen: list[int] = []
        total = 0
        for i in ordered:
            if total >= budget:
                break
            chosen.append(i)
            total += self._costs[i]
        return chosen

    def _order_bulk(self, chosen: list[int]) -> list[int]:
        """Arrange the big block so that no label comes in long streaks.

        A model that has just read twenty "N" answers in a row leans towards "N".
        We spread each group evenly along the block: the j-th of n items in a
        group is placed at position (j + offset) / n on a 0..1 line.
        """
        mode = self.config.bulk_order
        if mode == "selection" or self.config.selection == "first":
            return chosen
        rng = np.random.default_rng(self.config.seed + 1)
        if mode == "random":
            return [chosen[j] for j in rng.permutation(len(chosen))]
        by_group: dict[str, list[int]] = defaultdict(list)
        for i in chosen:
            by_group[self._group(i)].append(i)
        keyed: list[tuple[float, int]] = []
        for group in sorted(by_group):
            members = by_group[group]
            offset = float(rng.random())
            keyed.extend(((j + offset) / len(members), i) for j, i in enumerate(members))
        return [i for _, i in sorted(keyed)]

    # ---------------------------------------------------------------------- tail

    def nearest(self, text: str, k: int, same_stratum: bool = True) -> list[tuple[int, float]]:
        """Pool indices of the k most similar examples, best first."""
        view = self.spec.view(text)
        mask = self._strata.get(view.stratum) if same_stratum else None
        if mask is not None and int(mask.sum()) < k:
            mask = None  # too few examples in this group: search everywhere
        return self._bm25.top_k(self._features(view), k, mask=mask)

    def _tail(self, text: str) -> list[int]:
        cfg = self.config
        if cfg.tail_k <= 0:
            return []
        ranked = self.nearest(text, cfg.tail_k * 4)
        ranked = [(i, s) for i, s in ranked if s > 0] or ranked[: cfg.tail_k]
        if cfg.tail_mmr < 1.0 and self._vectors is not None and len(ranked) > cfg.tail_k:
            picked = mmr_select(
                [i for i, _ in ranked],
                [s for _, s in ranked],
                self._vectors,
                cfg.tail_k,
                cfg.tail_mmr,
            )
            score = dict(ranked)
            chosen = [(i, score[i]) for i in picked]
        else:
            chosen = ranked[: cfg.tail_k]

        if self.spec.kind == "classification" and cfg.tail_per_label > 0 and self.spec.labels:
            # Show the nearest examples of EVERY label, so the model sees where the
            # border between the labels runs for inputs like this one.
            have = Counter(self._labels[i] for i, _ in chosen)
            wide = self.nearest(text, len(self.pool))
            for label in self.spec.labels:
                need = cfg.tail_per_label - have.get(label, 0)
                if need <= 0:
                    continue
                in_tail = {i for i, _ in chosen}
                extra = [(i, s) for i, s in wide if self._labels[i] == label and i not in in_tail]
                chosen.extend(extra[:need])
        # Most similar example last, right next to the input it should inform.
        chosen.sort(key=lambda pair: (pair[1], -pair[0]))
        return [i for i, _ in chosen]

    # --------------------------------------------------------------------- build

    def route(self, text: str) -> int:
        if self.n_clusters == 1:
            return 0
        return int(self._route_features([self._features(self.spec.view(text))])[0])

    def bulk_examples(self, cluster: int = 0) -> list[Example]:
        return [self.pool[i] for i in self._global + self._cluster_blocks[cluster]]

    def prefix_messages(self, cluster: int = 0) -> list[dict[str, str]]:
        """Only the reusable beginning (used to warm the server cache)."""
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": self._bulk_text[cluster]},
        ]

    def with_guidelines(self, guidelines: str) -> ContextBuilder:
        """A copy that carries a rulebook in its system prompt.  Example selection,
        which is the expensive part, is shared with the original."""
        clone = copy.copy(self)
        clone.guidelines = guidelines
        clone._system = render.system_prompt(self.spec, self.definition, guidelines)
        clone._overhead = (
            self.counter.count(clone._system) + self.counter.count(render.BULK_HEADING) + 24
        )
        return clone

    def build(self, query: Query, session: Sequence[Example] = (), suffix: str = "") -> BuiltPrompt:
        """Assemble the prompt for one input.

        session  answers given earlier in this run (shown after the big block)
        suffix   extra text placed right after the input (task 8 uses it to pin
                 the exact function signature)
        """
        spec, cfg = self.spec, self.config
        cluster = self.route(query.input)
        tail = self._tail(query.input)
        query_text = render.render_input(query.input) + suffix

        session_text = ""
        if session:
            session_text = render.SESSION_HEADING + "".join(
                render.render_demo(spec, e) for e in session
            )
        session_tokens = self.counter.count(session_text) if session_text else 0

        reminder = f"{spec.answer_rule}\n\n" if cfg.restate_rule else ""
        head = self._bulk_text[cluster] + session_text
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system}]
        if cfg.tail_as_turns:
            # The nearest examples become real question/answer turns of the chat.
            # The model then sees "its own" earlier answers in exactly the format
            # we want next, which binds the output format more tightly than text.
            pending = head + render.TURNS_BRIDGE
            for i in tail:
                example = self.pool[i]
                messages.append(
                    {"role": "user", "content": pending + render.render_input(example.input)}
                )
                messages.append(
                    {"role": "assistant", "content": render.render_output(spec, example.output)}
                )
                pending = ""
            messages.append({"role": "user", "content": pending + reminder + query_text})
        else:
            tail_text = (
                render.TAIL_HEADING + "".join(self._rendered[i] for i in tail) if tail else ""
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"{head}{tail_text}# Your turn\n{reminder}{query_text}\nOutput:",
                }
            )

        tail_tokens = sum(self._costs[i] for i in tail)
        new_in_tail = sum(
            self._costs[i] for i in tail if self.pool[i].id not in self._bulk_ids[cluster]
        )
        bulk_tokens = self._bulk_tokens[cluster]
        per_turn = 8 * len(tail) if cfg.tail_as_turns else 0
        stats = ContextStats(
            bulk_tokens=bulk_tokens,
            session_tokens=session_tokens,
            tail_tokens=tail_tokens,
            unique_demo_tokens=bulk_tokens + new_in_tail,
            n_bulk=len(self._global) + len(self._cluster_blocks[cluster]),
            n_session=len(session),
            n_tail=len(tail),
            prompt_tokens=self._overhead
            + bulk_tokens
            + session_tokens
            + tail_tokens
            + per_turn
            + self.counter.count(query_text)
            + self.counter.count(reminder),
            min_required=self.min_required,
        )
        prefix_key = hashlib.sha256(f"{spec.task_id}:{cluster}:{head}".encode()).hexdigest()[:16]
        return BuiltPrompt(
            messages=messages,
            stats=stats,
            cluster=cluster,
            prefix_key=prefix_key,
            tail_ids=tuple(self.pool[i].id for i in tail),
        )
