"""The run journal: every finished answer is written to disk at once.

A full run is thousands of model calls and can take hours.  Machines get
restarted, sessions time out, servers crash.  If answers live only in memory,
a crash at 95% loses everything.

So each answer is appended to a file as one line of JSON the moment it exists.
After a crash, starting the same command again reads the file, skips what is
done, and continues.  Nothing is computed twice and nothing is lost.

Details that make this safe:
  * append-only: old lines are never rewritten, so a crash cannot damage them
  * flush + fsync after each line: the line is really on disk, not in a buffer
  * a half-written last line (crash mid-write) is ignored when reading
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from cairn.types import Prediction


class Journal:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Prediction]:
        """Read all finished answers.  A later line for the same id wins."""
        done: dict[str, Prediction] = {}
        if not self.path.exists():
            return done
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    prediction = Prediction.from_json(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # a line cut off by a crash; that input is simply redone
                done[prediction.query_id] = prediction
        return done

    def append(self, prediction: Prediction) -> None:
        line = json.dumps(prediction.to_json(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
