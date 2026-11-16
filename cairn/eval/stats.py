"""Is the difference real, or did I get lucky?

With 300 test inputs, "71% vs 73%" is 6 inputs.  Draw 300 different inputs and
the order could flip.  Three small tools keep us honest:

    wilson_interval   error bar for one accuracy number
    bootstrap_ci      error bar for any statistic, by re-drawing the data
    mcnemar_exact     "is method B really better than A on the SAME inputs?"

Rule used across the project: a change is adopted only when its error bars say
it is at least not worse.  A two-point gain inside the noise is not a gain.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% range for a success rate.  Behaves well near 0% and 100% and for small totals,
    where the simple "p plus or minus 2 sigma" formula gives nonsense."""
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,  # type: ignore[assignment]
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Error bar by "pretend the data is the world": draw the same number of items
    again and again with replacement and see how much the statistic moves."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(data), size=(n_resamples, len(data)))
    stats = np.array([statistic(data[row]) for row in picks])
    lower = (1.0 - confidence) / 2.0
    return float(np.quantile(stats, lower)), float(np.quantile(stats, 1.0 - lower))


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> tuple[int, int, float]:
    """Paired comparison of two methods on the same inputs.

    Only inputs where exactly one method is right carry information.  Call them
    b (only A right) and c (only B right).  If the methods were equally good,
    each such input would be a coin flip.  The p-value is the chance of a split
    at least this lopsided from fair coin flips.  Returns (b, c, p_value).
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("both methods must be scored on the same inputs")
    b = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return 0, 0, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, min(1.0, 2.0 * tail)
