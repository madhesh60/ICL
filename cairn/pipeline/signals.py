"""Building the row of cheap signals for one input (used by the stacked calibrator).

    nearest-examples vote   for each label: log of the similarity-weighted share
                            of that label among the 15 most similar pool examples
    task signals            whatever the task card measures (may be nothing)

The calibration examples are not in the pool, so the vote never sees its own
answer.  That matters: a signal that peeks at the answer during fitting looks
perfect, gets all the weight, and then fails on real inputs.
"""

from __future__ import annotations

import numpy as np

from cairn.context.builder import ContextBuilder
from cairn.eval.baselines import knn_distribution


def signal_row(builder: ContextBuilder, text: str, k: int = 15) -> np.ndarray:
    spec = builder.spec
    parts = [np.log(knn_distribution(builder, text, spec.labels, k=k))]
    if spec.signals is not None:
        parts.append(np.asarray(spec.signals(text), dtype=np.float64))
    return np.concatenate(parts)
