"""Self-consistency: ask several times, keep the answer that comes back most often.

A wrong answer can be wrong in many different ways, a right answer is right in
one way.  So when we sample five replies, mistakes scatter and the correct
answer piles up.  The share of replies that agree is also a useful, honest
confidence number.

Answers are cleaned before counting, otherwise "Paris", "paris." and " paris"
would split one vote into three.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cairn.decoding.parse import Normalizer


@dataclass(frozen=True)
class VoteResult:
    answer: str | None
    share: float  # winning votes / valid votes
    valid: int  # replies that could be parsed
    counts: dict[str, float]


def majority_vote(
    replies: Sequence[str],
    normalize: Normalizer,
    *,
    weights: Sequence[float] | None = None,
    allowed: Sequence[str] | None = None,
    group_key: Callable[[str], str] | None = None,
) -> VoteResult:
    """Count cleaned answers.  Ties go to the answer that appeared first.

    `allowed` drops answers outside a given set (used with conformal sets: only
    vote among the labels that are still plausible).

    `group_key` pools answers that mean the same thing but are written differently
    ("the beatles" / "beatles").  The group with the most votes wins, and the
    answer handed back is the most common *spelling* inside that group, so the
    output still looks like the examples.  Without pooling, five votes split
    3 + 2 between two spellings of the right answer can lose to a wrong answer
    with 4 votes.
    """
    counts: dict[str, float] = {}
    valid = 0
    for index, reply in enumerate(replies):
        answer = normalize(reply)
        if answer is None or (allowed is not None and answer not in allowed):
            continue
        valid += 1
        counts[answer] = counts.get(answer, 0.0) + (weights[index] if weights else 1.0)
    if not counts:
        return VoteResult(answer=None, share=0.0, valid=0, counts={})
    total = sum(counts.values())
    if group_key is None:
        winner = max(counts, key=lambda a: counts[a])  # max() keeps the first of equal bests
        return VoteResult(answer=winner, share=counts[winner] / total, valid=valid, counts=counts)
    groups: dict[str, float] = {}
    for answer, count in counts.items():
        key = group_key(answer)
        groups[key] = groups.get(key, 0.0) + count
    best_group = max(groups, key=lambda g: groups[g])
    members = [a for a in counts if group_key(a) == best_group]
    winner = max(members, key=lambda a: counts[a])
    return VoteResult(answer=winner, share=groups[best_group] / total, valid=valid, counts=counts)
