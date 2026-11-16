"""Remember every answer on disk, so the same question is never paid for twice.

Why this matters more than it looks:

* A crashed run restarts in seconds, because everything already answered is
  read back from the file.
* Changing one late step (say, the vote rule) does not re-run the expensive
  early steps.
* Results become reproducible: the same inputs give byte-identical outputs.

Only requests that are deterministic are stored (temperature 0, or a fixed
seed).  A request that is *meant* to be random must not be frozen.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from cairn.llm.base import GenRequest, GenResponse, LLMBackend


class CachedBackend:
    def __init__(self, inner: LLMBackend, path: Path | str) -> None:
        self.inner = inner
        self.model = inner.model
        self.hits = 0
        self.misses = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by all worker threads, guarded by a lock.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()
        self._lock = threading.Lock()

    def generate(self, request: GenRequest) -> GenResponse:
        if not request.deterministic:
            return self.inner.generate(request)
        key = request.cache_key(self.model)
        with self._lock:
            row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            response = GenResponse.from_json(json.loads(row[0]))
            response.from_cache = True
            return response
        self.misses += 1
        response = self.inner.generate(request)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
                (key, json.dumps(response.to_json(), ensure_ascii=False)),
            )
            self._db.commit()
        return response

    def close(self) -> None:
        with self._lock:
            self._db.close()
