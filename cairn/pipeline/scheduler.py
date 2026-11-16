"""Cache-aware scheduling: the order in which requests are sent matters.

The server keeps "notes" (a KV cache) for prompt beginnings it has already
read.  Two things can waste that:

1. Mixing groups.  If prompts with beginning A and beginning B alternate, and
   the server's memory only holds a few beginnings, it keeps throwing notes
   away and re-reading 30,000 tokens.  Fix: send all of A, then all of B.

2. The stampede.  If 16 workers send the first 16 prompts of a group at the
   same instant, none of them finds notes (nobody has finished reading yet),
   and the server reads the same 30,000 tokens 16 times in parallel.
   Fix: send ONE request first (the "primer"), wait for it, then let the
   other workers go.  They all find the notes.

`run_grouped` does exactly that and returns results in the original order.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_grouped(
    items: Sequence[T],
    key: Callable[[T], Hashable],
    work: Callable[[T], R],
    *,
    workers: int = 8,
    on_result: Callable[[T, R], None] | None = None,
) -> list[R]:
    """Run `work` on every item: group by `key`, prime each group, then fan out.

    An exception inside `work` stops the run and is raised here.  Work that
    should survive a bad item must catch its own errors and return a fallback.
    """
    groups: dict[Hashable, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(key(item), []).append(index)

    results: list[R | None] = [None] * len(items)

    def do(index: int) -> None:
        results[index] = work(items[index])
        if on_result is not None:
            on_result(items[index], results[index])  # type: ignore[arg-type]

    # Biggest group first: if the run is interrupted, the most work is already saved.
    for indices in sorted(groups.values(), key=len, reverse=True):
        do(indices[0])  # the primer, alone
        rest = indices[1:]
        if not rest:
            continue
        if workers <= 1:
            for index in rest:
                do(index)
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(do, index) for index in rest]:
                future.result()  # re-raises the first failure
    return results  # type: ignore[return-value]
