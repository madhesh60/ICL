"""Scoring a set of predictions against known answers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from cairn.decoding.calibrate import expected_calibration_error
from cairn.eval.stats import wilson_interval
from cairn.types import Prediction, Query


@dataclass
class EvalReport:
    n: int
    correct: int
    accuracy: float
    ci_low: float
    ci_high: float
    ece: float  # how honest the confidence numbers are (0 = perfectly honest)
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)
    hits: list[bool] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"accuracy {self.accuracy:.1%} ({self.correct}/{self.n}, "
            f"95% range {self.ci_low:.1%}..{self.ci_high:.1%}), ECE {self.ece:.3f}"
        )

    def to_json(self) -> dict[str, object]:
        return {
            "n": self.n,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "ece": round(self.ece, 4),
            "by_source": self.by_source,
        }


def is_correct(answer: str, gold: str) -> bool:
    """Exact match after trimming outer spaces, as the official metric describes."""
    return answer.strip() == gold.strip()


def evaluate(predictions: Sequence[Prediction], queries: Sequence[Query]) -> EvalReport:
    """Accuracy with an error bar, plus a breakdown by the stage that answered.

    The breakdown is the first thing to read when a number disappoints: it shows
    whether mistakes come from the cheap stage, the expensive stage or the fallback.
    """
    gold = {q.id: q.gold for q in queries}
    hits: list[bool] = []
    confidence: list[float] = []
    per_source: dict[str, list[bool]] = defaultdict(list)
    for prediction in predictions:
        answer = gold.get(prediction.query_id)
        if answer is None:
            raise ValueError(
                f"no gold answer for {prediction.query_id}; evaluate needs labelled queries"
            )
        hit = is_correct(prediction.answer, answer)
        hits.append(hit)
        confidence.append(prediction.confidence)
        per_source[prediction.source].append(hit)
    n, correct = len(hits), sum(hits)
    low, high = wilson_interval(correct, n)
    return EvalReport(
        n=n,
        correct=correct,
        accuracy=correct / n if n else 0.0,
        ci_low=low,
        ci_high=high,
        ece=expected_calibration_error(confidence, hits),
        by_source={
            source: {
                "n": len(h),
                "share": round(len(h) / n, 4),
                "accuracy": round(sum(h) / len(h), 4),
            }
            for source, h in sorted(per_source.items())
        },
        hits=hits,
    )
