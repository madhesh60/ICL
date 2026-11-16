"""Fixing a model that leans towards some labels for no good reason.

A language model shown 600 examples, 67% of them "N", will say "N" too often.
It also prefers labels that are common words, and the label of the last example
it read.  None of that has anything to do with the input.  Calibration measures
the lean and subtracts it.

Three calibrators, from "needs nothing" to "needs labelled data":

    BatchCalibrator        needs only the scores of the inputs we are labelling.
                           Idea: averaged over many inputs, each label's score
                           should not stand out; subtract the average.
    ContextualCalibrator   needs scores for a few empty inputs ("N/A").  Whatever
                           the model answers for an empty input is pure lean.
    VectorScaling          needs labelled held-out examples.  Learns one number
                           per label (and one overall sharpness) that makes the
                           scores match the truth best.

We have hundreds of labelled examples left over that did not fit in the prompt,
so VectorScaling is the default.  The other two are kept for comparison and for
situations with no labels to spare.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class IdentityCalibrator:
    def transform(self, logp: np.ndarray) -> np.ndarray:
        return _log_softmax(np.atleast_2d(logp))


class BatchCalibrator:
    """Subtract each label's mean log-score over a batch (Zhou et al., 2023)."""

    def __init__(self) -> None:
        self.bias: np.ndarray | None = None

    def fit(self, logp: np.ndarray) -> BatchCalibrator:
        self.bias = np.atleast_2d(logp).mean(axis=0)
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        if self.bias is None:
            raise RuntimeError("fit() first")
        return _log_softmax(np.atleast_2d(logp) - self.bias)


class ContextualCalibrator:
    """Divide by what the model says for content-free inputs (Zhao et al., 2021)."""

    def __init__(self) -> None:
        self.bias: np.ndarray | None = None

    def fit(self, content_free_logp: np.ndarray) -> ContextualCalibrator:
        probs = np.exp(_log_softmax(np.atleast_2d(content_free_logp))).mean(axis=0)
        self.bias = np.log(np.maximum(probs, 1e-9))
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        if self.bias is None:
            raise RuntimeError("fit() first")
        return _log_softmax(np.atleast_2d(logp) - self.bias)


class VectorScaling:
    """Learn `scale * logp + bias` on labelled data by minimising the log loss.

    This is a tiny logistic regression with K+1 numbers.  The loss is bowl-shaped
    (convex) in those numbers, so plain gradient steps reach the one best answer;
    no random start, no luck involved.  A small pull towards "change nothing"
    (`l2`) keeps it from over-reacting when there are few examples.
    """

    def __init__(self, l2: float = 1e-2, steps: int = 400, lr: float = 0.5) -> None:
        self.l2 = l2
        self.steps = steps
        self.lr = lr
        self.scale = 1.0
        self.bias: np.ndarray | None = None

    def fit(self, logp: np.ndarray, labels: Sequence[int]) -> VectorScaling:
        x = np.atleast_2d(np.asarray(logp, dtype=np.float64))
        y = np.asarray(labels, dtype=np.int64)
        n, k = x.shape
        onehot = np.eye(k)[y]
        scale, bias = 1.0, np.zeros(k)
        for _ in range(self.steps):
            probs = np.exp(_log_softmax(scale * x + bias))
            diff = (probs - onehot) / n
            grad_bias = diff.sum(axis=0) + self.l2 * bias
            grad_scale = float((diff * x).sum()) + self.l2 * (scale - 1.0)
            bias -= self.lr * grad_bias
            scale = max(0.05, scale - self.lr * grad_scale)
        self.scale, self.bias = float(scale), bias - bias.mean()
        return self

    def transform(self, logp: np.ndarray) -> np.ndarray:
        if self.bias is None:
            raise RuntimeError("fit() first")
        return _log_softmax(self.scale * np.atleast_2d(logp) + self.bias)


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """How far "I am 80% sure" is from "I am right 80% of the time", averaged over bins."""
    conf = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(error)
