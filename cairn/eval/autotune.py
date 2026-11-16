"""Choosing between pipeline settings by racing them, not by guessing.

We have labelled held-out examples, so "is setting A better than B?" has a
measurable answer.  Measuring every candidate on all 300 dev examples is
wasteful though: a clearly bad candidate shows itself after 40.

Successive halving (the core of the Hyperband method) races candidates:

    round 1: every candidate labels the first 40 dev examples
             -> drop the worse half
    round 2: the survivors label 80 examples
             -> drop the worse half
    ...until one is left or the dev set is used up.

Most of the compute goes to the candidates that might actually win.  Because
each round re-uses the same first examples and answers are cached on disk, a
survivor never pays twice for an example it already labelled.

Ties are broken by cost (fewer model calls wins): if two settings are equally
accurate, the cheaper one is the better engineering choice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from cairn.eval.metrics import EvalReport, evaluate
from cairn.eval.stats import mcnemar_exact
from cairn.pipeline.runner import TaskRunner
from cairn.types import Query

log = logging.getLogger(__name__)


@dataclass
class Trial:
    name: str
    rounds: list[dict[str, float]] = field(default_factory=list)
    final: EvalReport | None = None
    calls: int = 0


@dataclass
class TuneResult:
    winner: str
    trials: dict[str, Trial]
    # Paired test of the winner against the runner-up on the largest shared sample.
    runner_up: str | None = None
    p_value: float | None = None
    compared_on: int = 0  # examples both labelled

    def table(self) -> str:
        lines = ["| setting | examples seen | accuracy | model calls |", "|---|---|---|---|"]
        ranked = sorted(
            self.trials.values(),
            key=lambda t: (-len(t.rounds), -(t.final.accuracy if t.final else 0)),
        )
        for trial in ranked:
            last = trial.rounds[-1]
            mark = " **(winner)**" if trial.name == self.winner else ""
            lines.append(
                f"| {trial.name}{mark} | {int(last['n'])} | {last['accuracy']:.1%} | {trial.calls} |"
            )
        return "\n".join(lines)


def successive_halving(
    runners: dict[str, Callable[[], TaskRunner]],
    dev: Sequence[Query],
    *,
    start: int = 40,
    keep: float = 0.5,
) -> TuneResult:
    """Race named runner factories on `dev`; returns the winner and the full record."""
    if not runners:
        raise ValueError("nothing to tune: no candidates given")
    alive: dict[str, TaskRunner] = {}
    trials: dict[str, Trial] = {name: Trial(name) for name in runners}
    for name, make in runners.items():
        runner = make()
        # Prepare once on the whole dev set so the prompt does not change between rounds.
        runner.prepare(dev)
        alive[name] = runner

    n = min(start, len(dev))
    while True:
        scored: list[tuple[float, int, str]] = []
        for name, runner in alive.items():
            batch = dev[:n]
            report = evaluate(runner.run(batch), batch)
            trial = trials[name]
            trial.final = report
            trial.calls = runner.meter.total_calls()
            trial.rounds.append({"n": n, "accuracy": report.accuracy})
            scored.append((-report.accuracy, trial.calls, name))
            log.info("tune: %-24s n=%d %s", name, n, report.line())
        scored.sort()
        if len(alive) == 1 or n >= len(dev):
            break
        survivors = max(1, int(len(scored) * keep))
        for _, _, name in scored[survivors:]:
            del alive[name]
        n = min(len(dev), n * 2)

    winner = scored[0][2]
    result = TuneResult(winner=winner, trials=trials)
    # Runner-up: the candidate that survived longest after the winner, best accuracy first.
    ranked = sorted(
        (t for t in trials.values() if t.name != winner and t.final is not None),
        key=lambda t: (-len(t.rounds), -t.final.accuracy, t.calls),  # type: ignore[union-attr]
    )
    if ranked:
        second = ranked[0].name
        first_hits = trials[winner].final.hits  # type: ignore[union-attr]
        second_hits = trials[second].final.hits  # type: ignore[union-attr]
        # Both labelled the same first examples of `dev`, so compare them on that shared part.
        shared = min(len(first_hits), len(second_hits))
        result.runner_up = second
        result.p_value = mcnemar_exact(second_hits[:shared], first_hits[:shared])[2]
        result.compared_on = shared
    return result
