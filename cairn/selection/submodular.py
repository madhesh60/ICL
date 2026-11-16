"""Packing the most information into a fixed number of tokens.

The picture: you must fill a 30K-token shelf with examples.  Each example brings
some "new things" (words, patterns, a label) and takes some shelf space (tokens).
The first example with the word "funeral" teaches a lot; the tenth one teaches
almost nothing new.  That "each extra copy helps less" shape is called
*diminishing returns* (the maths name is "submodular").

We score a set S of examples as

    value(S) = sum over features f of   weight_f * sqrt( how often f appears in S )
             + sum over chosen examples of bonus_i

The square root is what creates diminishing returns.  For this kind of score a
simple greedy rule ("always take the example with the best gain per token") is
provably close to the best possible set, and the "lazy" trick below makes it fast.

Quotas keep groups (labels, question types, genres) in a chosen proportion, so
a majority label cannot crowd out the others.
"""

from __future__ import annotations

import heapq
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoverageResult:
    order: list[int]  # chosen example indices, in the order greedy picked them
    total_cost: int
    value: float
    group_cost: dict[Hashable, int]


def _gain(ids: np.ndarray, vals: np.ndarray, weights: np.ndarray, mass: np.ndarray) -> float:
    return float(np.dot(weights[ids], np.sqrt(mass[ids] + vals) - np.sqrt(mass[ids])))


def coverage_value(
    chosen: Sequence[int],
    features: Sequence[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    bonus: np.ndarray | None = None,
) -> float:
    """Score of a set under the same objective greedy optimises (for comparisons)."""
    mass = np.zeros(len(weights), dtype=np.float64)
    for i in chosen:
        ids, vals = features[i]
        np.add.at(mass, ids, vals)
    value = float(np.dot(weights, np.sqrt(mass)))
    if bonus is not None:
        value += float(sum(bonus[i] for i in chosen))
    return value


def budgeted_coverage(
    features: Sequence[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    costs: Sequence[int],
    min_total_cost: int,
    *,
    bonus: np.ndarray | None = None,
    groups: Sequence[Hashable] | None = None,
    group_share: Mapping[Hashable, float] | None = None,
    candidates: Sequence[int] | None = None,
    already_chosen: Sequence[int] = (),
) -> CoverageResult:
    """Greedily pick examples until their token cost reaches `min_total_cost`.

    features[i]   (feature ids, feature values) of example i; ids must be unique
    weights       importance of each feature id (rare features weigh more)
    costs[i]      token cost of example i
    bonus[i]      extra value for picking i regardless of the others (e.g. relevance
                  to a cluster); adding it keeps the diminishing-returns property
    groups[i]     group key of example i; with `group_share` it caps each group at
                  its share of the budget
    candidates    restrict the choice to these indices (default: all)
    already_chosen  examples that are in the prompt anyway (a shared block).  They
                  cost nothing here and are never picked again, but what they
                  already teach counts, so the new picks complement them.

    Lazy greedy: a gain computed earlier can only have shrunk since (diminishing
    returns), so it is an upper bound.  We keep bounds in a heap, re-check only
    the top one, and take it if it is still on top.  Most examples are never
    re-scored, which turns minutes into milliseconds.
    """
    n = len(features)
    if len(costs) != n:
        raise ValueError("features and costs must have the same length")
    taken = set(already_chosen)
    pool = [i for i in (range(n) if candidates is None else candidates) if i not in taken]
    mass = np.zeros(len(weights), dtype=np.float64)
    for i in taken:
        mass[features[i][0]] += features[i][1]
    extra = np.zeros(n, dtype=np.float64) if bonus is None else np.asarray(bonus, dtype=np.float64)

    quota: dict[Hashable, float] | None = None
    if groups is not None and group_share is not None:
        quota = {g: share * min_total_cost for g, share in group_share.items()}
    used: dict[Hashable, int] = {}

    # Heap entries: (-gain per token, index, round in which the gain was computed).
    heap: list[tuple[float, int, int]] = []
    for i in pool:
        if costs[i] <= 0:
            continue
        ids, vals = features[i]
        heap.append((-(_gain(ids, vals, weights, mass) + extra[i]) / costs[i], i, 0))
    heapq.heapify(heap)

    order: list[int] = []
    total = 0
    value = 0.0
    skipped: list[int] = []  # blocked by a full quota; used only if we run short
    while heap and total < min_total_cost:
        neg_ratio, i, stamp = heapq.heappop(heap)
        if quota is not None:
            g = groups[i]  # type: ignore[index]
            if used.get(g, 0) >= quota.get(g, 0.0):
                skipped.append(i)
                continue
        ids, vals = features[i]
        if stamp != len(order):
            fresh = _gain(ids, vals, weights, mass) + extra[i]
            heapq.heappush(heap, (-fresh / costs[i], i, len(order)))
            continue
        value += -neg_ratio * costs[i]
        mass[ids] += vals
        order.append(i)
        total += costs[i]
        if groups is not None:
            used[groups[i]] = used.get(groups[i], 0) + costs[i]

    # Quotas are a preference, the token minimum is a rule.  If the quotas left us
    # short (a rare group ran out of examples), fill up from what was skipped.
    for i in skipped:
        if total >= min_total_cost:
            break
        ids, vals = features[i]
        value += _gain(ids, vals, weights, mass) + extra[i]
        mass[ids] += vals
        order.append(i)
        total += costs[i]
        if groups is not None:
            used[groups[i]] = used.get(groups[i], 0) + costs[i]

    return CoverageResult(order=order, total_cost=total, value=value, group_cost=used)
