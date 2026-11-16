"""Session memory for continuous annotation.

When annotation runs for a long time, two things are wanted:

    consistency  - the same input must get the same answer every time
    learning     - sure answers given earlier can help with later inputs

Both have to work WITHOUT destroying the server's prompt cache.  The rules:

1. Exact repeats are answered from memory.  No model call, no chance of
   answering differently.

2. Sure answers are added to a "session block" that sits after the big shared
   example block.  The block is APPEND-ONLY: new entries go at the end, old
   entries never move or change.  The prompt beginning the server has cached
   stays valid; it only has to read the few new lines.

3. The block changes only between waves of work, never in the middle of one.
   Inside a wave every prompt sees the same block, so they all share one cache
   entry even when many workers run at once.

4. Admission is strict, because a wrong answer in the block would be copied by
   later answers (errors feeding on themselves).  An answer gets in only if its
   confidence is high, and no single label may fill more than its share.
"""

from __future__ import annotations

import re
import threading
from collections import Counter

from cairn.types import Example, Prediction, Query


def _fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class AnnotationMemory:
    def __init__(
        self,
        *,
        max_items: int = 48,
        min_confidence: float = 0.95,
        max_label_share: float = 0.6,
        trusted_sources: tuple[str, ...] = ("score", "think_vote", "program", "generate"),
    ) -> None:
        self.max_items = max_items
        self.min_confidence = min_confidence
        self.max_label_share = max_label_share
        self.trusted_sources = trusted_sources
        self._answers: dict[str, Prediction] = {}
        self._committed: list[Example] = []  # what prompts currently see
        self._waiting: list[Example] = []  # admitted, shown from the next wave on
        self._lock = threading.Lock()

    def recall(self, query: Query) -> Prediction | None:
        """The earlier answer for an identical input, if there is one."""
        with self._lock:
            hit = self._answers.get(_fingerprint(query.input))
        if hit is None:
            return None
        return Prediction(query.id, hit.answer, hit.confidence, "memory", {"same_as": hit.query_id})

    def record(self, query: Query, prediction: Prediction) -> bool:
        """Store an answer.  Returns True if it was also admitted to the session block."""
        with self._lock:
            self._answers.setdefault(_fingerprint(query.input), prediction)
            if (
                prediction.confidence < self.min_confidence
                or prediction.source not in self.trusted_sources
                or len(self._committed) + len(self._waiting) >= self.max_items
            ):
                return False
            counts = Counter(e.output for e in self._committed + self._waiting)
            total = sum(counts.values())
            # Let the first few in freely; after that no label may dominate the block.
            if total >= 4 and (counts[prediction.answer] + 1) / (total + 1) > self.max_label_share:
                return False
            self._waiting.append(Example(query.id, query.input, prediction.answer))
            return True

    def commit(self) -> int:
        """End of a wave: make the waiting entries visible.  Returns how many were added."""
        with self._lock:
            added = len(self._waiting)
            self._committed.extend(self._waiting)
            self._waiting.clear()
            return added

    def session(self) -> tuple[Example, ...]:
        with self._lock:
            return tuple(self._committed)
