"""Reference points that need no language model.

Before trusting any model score, know what "no intelligence at all" scores:

    majority   always answer the most common label
    knn_vote   answer like the most similar labelled examples (word overlap only)

If the model pipeline does not beat these, the model is not adding anything and
something is broken.  If kNN alone is already strong on a task, that tells us
the retrieved examples carry real signal there, so the per-input tail matters.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from cairn.context.builder import ContextBuilder
from cairn.eval.metrics import is_correct
from cairn.types import Example, Query


def majority_accuracy(pool: Sequence[Example], queries: Sequence[Query]) -> tuple[str, float]:
    label, _ = Counter(e.output for e in pool).most_common(1)[0]
    hits = [is_correct(label, q.gold or "") for q in queries]
    return label, float(np.mean(hits)) if hits else 0.0


def knn_distribution(
    builder: ContextBuilder, text: str, labels: Sequence[str], k: int = 15, smoothing: float = 0.5
) -> np.ndarray:
    """Label probabilities from the k nearest pool examples, weighted by similarity."""
    votes = np.full(len(labels), smoothing, dtype=np.float64)
    index = {label: i for i, label in enumerate(labels)}
    for i, score in builder.nearest(text, k):
        label = builder.pool[i].output
        if label in index:
            votes[index[label]] += max(score, 0.0)
    return votes / votes.sum()


def knn_accuracy(builder: ContextBuilder, queries: Sequence[Query], k: int = 15) -> float:
    """Accuracy of answering with the similarity-weighted vote of the k nearest examples.

    For tasks with open answers (trivia) this is "copy the answer of a similar
    question", which is almost always wrong; the number is still a useful floor.
    """
    hits = []
    for q in queries:
        votes: dict[str, float] = {}
        for i, score in builder.nearest(q.input, k):
            label = builder.pool[i].output
            votes[label] = votes.get(label, 0.0) + max(score, 1e-9)
        answer = max(votes, key=lambda a: votes[a]) if votes else ""
        hits.append(is_correct(answer, q.gold or ""))
    return float(np.mean(hits)) if hits else 0.0
