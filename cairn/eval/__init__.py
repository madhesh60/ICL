"""Measuring: scores, error bars, fair comparisons, and model-free reference points."""

from cairn.eval.metrics import EvalReport, evaluate
from cairn.eval.stats import bootstrap_ci, mcnemar_exact, wilson_interval

__all__ = ["EvalReport", "bootstrap_ci", "evaluate", "mcnemar_exact", "wilson_interval"]
