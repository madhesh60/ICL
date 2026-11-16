"""Loading the task files and cutting the labelled examples into roles.

Why roles?  A task gives us thousands of labelled examples but only a few hundred
fit in a prompt.  We cut the examples into four groups that never overlap:

    pool     - the only examples that may appear inside a prompt
    calib    - used to fit calibration and conformal thresholds, and to check
               induced programs ("does this program reproduce these answers?")
    dev      - used to compare strategies and to tune settings
    holdout  - touched once, at the very end, to get an honest final number

If one example sat in both `pool` and `dev`, the model could simply copy the
answer out of its prompt and our dev score would be a lie.  The split below is
made by hashing each example id, so it does not depend on file order and it is
identical on every machine.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from cairn.types import Example, Query

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = (
    REPO_ROOT / "OpenSeek" / "openseek" / "competition" / "LongContext-ICL-Annotation" / "data"
)

TASK_FILES: dict[int, str] = {
    1: "openseek-1_closest_integers.json",
    2: "openseek-2_count_nouns_verbs.json",
    3: "openseek-3_collatz_conjecture.json",
    4: "openseek-4_conala_concat_strings.json",
    5: "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    6: "openseek-6_mnli_same_genre_classification.json",
    7: "openseek-7_jeopardy_answer_generation_all.json",
    8: "openseek-8_kernel_generation.json",
}


class DataError(ValueError):
    """The task file is missing or does not have the shape we rely on."""


@dataclass(frozen=True)
class TaskData:
    task_id: int
    name: str
    definition: str
    examples: tuple[Example, ...]
    test: tuple[Query, ...]


@dataclass(frozen=True)
class SplitSizes:
    calib: int = 300
    dev: int = 300
    holdout: int = 200
    # Never reserve more than this share of the examples; small tasks (task 8 has
    # only 184 examples) scale the three numbers down together.
    max_reserved_frac: float = 0.35


@dataclass(frozen=True)
class Splits:
    pool: tuple[Example, ...]
    calib: tuple[Example, ...]
    dev: tuple[Example, ...]
    holdout: tuple[Example, ...]

    def as_queries(self, role: str) -> list[Query]:
        """Turn a labelled role into queries that carry their gold answer."""
        examples: tuple[Example, ...] = getattr(self, role)
        return [Query(id=e.id, input=e.input, gold=e.output) for e in examples]


def data_dir() -> Path:
    return Path(os.environ.get("CAIRN_DATA_DIR", DEFAULT_DATA_DIR))


def load_task(task_id: int, directory: Path | str | None = None) -> TaskData:
    """Read one task file and check everything we later take for granted."""
    if task_id not in TASK_FILES:
        raise DataError(f"task_id must be 1..8, got {task_id}")
    path = Path(directory or data_dir()) / TASK_FILES[task_id]
    if not path.exists():
        raise DataError(
            f"task file not found: {path}\n"
            "Set CAIRN_DATA_DIR or pass --data-dir to point at the folder with the 8 json files."
        )
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    try:
        definition = raw["Definition"][0]
        examples = tuple(
            Example(id=str(e["id"]), input=str(e["input"]), output=str(e["output"][0]))
            for e in raw["examples"]
        )
        test = tuple(Query(id=str(t["id"]), input=str(t["input"])) for t in raw["test_samples"])
    except (KeyError, IndexError, TypeError) as err:
        raise DataError(f"{path.name} does not have the expected fields: {err!r}") from err

    _check_unique([e.id for e in examples], f"{path.name} examples")
    _check_unique([t.id for t in test], f"{path.name} test_samples")
    return TaskData(
        task_id=task_id,
        name=str(raw.get("task_name", f"openseek-{task_id}")),
        definition=definition,
        examples=examples,
        test=test,
    )


def _check_unique(ids: list[str], what: str) -> None:
    if len(set(ids)) != len(ids):
        raise DataError(f"duplicate ids in {what}")


def _stable_rank(example_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def split_examples(
    examples: tuple[Example, ...] | list[Example],
    sizes: SplitSizes | None = None,
    seed: int = 42,
) -> Splits:
    """Cut the examples into pool / calib / dev / holdout.

    Each example gets a pseudo-random rank from a hash of (seed, id).  We sort by
    that rank and slice.  Adding or removing other examples never moves an
    example that stays, which keeps old experiment results comparable.
    """
    sizes = sizes or SplitSizes()
    ordered = sorted(examples, key=lambda e: _stable_rank(e.id, seed))
    n = len(ordered)
    wanted = [sizes.calib, sizes.dev, sizes.holdout]
    cap = int(n * sizes.max_reserved_frac)
    if sum(wanted) > cap and sum(wanted) > 0:
        scale = cap / sum(wanted)
        wanted = [int(w * scale) for w in wanted]
    n_calib, n_dev, n_holdout = wanted
    calib = ordered[:n_calib]
    dev = ordered[n_calib : n_calib + n_dev]
    holdout = ordered[n_calib + n_dev : n_calib + n_dev + n_holdout]
    pool = ordered[n_calib + n_dev + n_holdout :]
    # Give the pool back its original file order: reproducible and easy to inspect.
    position = {e.id: i for i, e in enumerate(examples)}
    pool.sort(key=lambda e: position[e.id])
    return Splits(pool=tuple(pool), calib=tuple(calib), dev=tuple(dev), holdout=tuple(holdout))
