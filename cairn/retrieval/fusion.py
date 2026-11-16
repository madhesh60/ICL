"""Combining several rankings, and picking results that are not all the same."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge rankings by position, not by raw score.

    Two search methods give scores on different scales, so adding scores is
    meaningless.  Positions are comparable: an item gets 1/(k+rank) from each
    list it appears in.  Items that several methods like float to the top.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


def mmr_select(
    candidates: Sequence[int],
    relevance: Sequence[float],
    vectors: np.ndarray,
    k: int,
    trade_off: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance: pick relevant items that differ from each other.

    Each step takes the candidate with the best
        trade_off * relevance  -  (1 - trade_off) * similarity to what we already picked
    so five near-copies of one example cannot fill five slots.
    """
    if not candidates:
        return []
    rel = np.asarray(relevance, dtype=np.float32)
    spread = float(rel.max() - rel.min())
    rel = (rel - rel.min()) / spread if spread > 0 else np.ones_like(rel)
    cand_vecs = vectors[np.asarray(candidates)]
    chosen: list[int] = []
    max_sim = np.zeros(len(candidates), dtype=np.float32)
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(min(k, len(candidates))):
        gain = trade_off * rel - (1.0 - trade_off) * max_sim
        gain[~available] = -np.inf
        best = int(np.argmax(gain))
        chosen.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, cand_vecs @ cand_vecs[best])
    return [candidates[i] for i in chosen]
