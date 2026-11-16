"""The three small records that travel through the whole system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Example:
    """A labelled demonstration: an input together with its correct output."""

    id: str
    input: str
    output: str


@dataclass(frozen=True, slots=True)
class Query:
    """An input we must label. `gold` is only known on our own dev split."""

    id: str
    input: str
    gold: str | None = None


@dataclass(slots=True)
class Prediction:
    """The answer for one query plus where it came from and how sure we are.

    `source` names the stage that produced the answer ("score", "think_vote",
    "program", "fallback_knn", ...). Keeping it lets us answer, after a run,
    "which stage is responsible for the mistakes?" without re-running anything.
    """

    query_id: str
    answer: str
    confidence: float = 0.0
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "confidence": round(float(self.confidence), 6),
            "source": self.source,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Prediction:
        return cls(
            query_id=row["query_id"],
            answer=row["answer"],
            confidence=float(row.get("confidence", 0.0)),
            source=row.get("source", ""),
            meta=dict(row.get("meta", {})),
        )
