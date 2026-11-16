"""Conformal prediction: turning "the model's confidence" into a promise we can check.

Problem: a model that says "90% sure" may be right only 70% of the time, so a
rule like "trust answers above 0.9" has no real meaning.

Conformal prediction fixes that with held-out labelled examples.  Instead of
one answer it returns a *set* of answers, built so that

    the true answer is inside the set at least (1 - alpha) of the time.

How:  on the held-out examples, look at the probability the model gave to the
TRUE label.  Find the level `q` such that the true label's score clears it
(1 - alpha) of the time.  For a new input, keep every label whose score clears
`q`.  That is all.  The promise holds for any model, good or bad, as long as
the new inputs come from the same source as the held-out ones.

How Cairn uses the set:

    one label in the set   -> the cheap answer is trusted, we stop here
    several labels         -> the model is honestly unsure: spend more compute
                              (thinking + voting), and only among those labels
    empty set              -> very unusual input: also escalate

So the size of the set decides where the expensive compute goes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalReport:
    coverage: float  # share of inputs whose true label is in the set
    mean_set_size: float
    singleton_rate: float  # share of inputs answered with exactly one label
    singleton_accuracy: float  # accuracy on those


class SplitConformal:
    """Split conformal classification with the "1 - probability of the true label" score."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.threshold: float | None = None

    def fit(self, probs: np.ndarray, labels: Sequence[int]) -> SplitConformal:
        probs = np.atleast_2d(probs)
        y = np.asarray(labels, dtype=np.int64)
        n = len(y)
        if n == 0:
            raise ValueError("need at least one calibration example")
        scores = 1.0 - probs[np.arange(n), y]
        # The (n + 1) is the finite-sample correction that makes the promise exact
        # rather than "roughly true for large n".
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        self.threshold = 1.0 if rank > n else float(np.sort(scores)[rank - 1])
        return self

    def predict_sets(self, probs: np.ndarray) -> list[list[int]]:
        if self.threshold is None:
            raise RuntimeError("fit() first")
        probs = np.atleast_2d(probs)
        keep = (1.0 - probs) <= self.threshold + 1e-12
        return [list(np.flatnonzero(row)) for row in keep]

    def report(self, probs: np.ndarray, labels: Sequence[int]) -> ConformalReport:
        sets = self.predict_sets(probs)
        y = list(labels)
        covered = [label in s for s, label in zip(sets, y, strict=True)]
        singles = [(s[0] == label) for s, label in zip(sets, y, strict=True) if len(s) == 1]
        return ConformalReport(
            coverage=float(np.mean(covered)),
            mean_set_size=float(np.mean([len(s) for s in sets])),
            singleton_rate=len(singles) / len(sets),
            singleton_accuracy=float(np.mean(singles)) if singles else 0.0,
        )
