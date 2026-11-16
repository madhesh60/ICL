"""TaskSpec: everything the engine needs to know about one task.

The engine itself has no `if task_id == 5` anywhere.  Each task hands the engine
a small card that answers five questions:

    kind        - what sort of problem is it?
    view        - which part of the input matters for "find similar examples"?
    normalize   - how do I clean a raw reply into a valid answer?
    constraint  - what shape must an answer have?
    answer_rule - one sentence telling the model the output format

Adding a ninth task means writing one more card, not touching the engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from cairn.decoding.constraints import Constraint
from cairn.decoding.parse import Normalizer

TaskKind = Literal["classification", "deterministic", "generation", "code"]


@dataclass(frozen=True)
class View:
    """The input as the retriever should see it.

    text     - words used for matching
    stratum  - a hard group; similar-example search stays inside it
               (question type for task 2, genre for task 6)
    boosted  - words that count double (the category in a trivia clue)
    """

    text: str
    stratum: str = ""
    boosted: str = ""


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    kind: TaskKind
    min_context_tokens: int
    view: Callable[[str], View]
    normalize: Normalizer
    answer_rule: str
    constraint: Constraint | None = None
    # Closed label set, filled from the data at run time for classification tasks.
    labels: tuple[str, ...] = ()
    # How the answer is rendered inside an example ("plain" or a fenced code block).
    output_style: Literal["plain", "code"] = "plain"
    # Tokens the model may spend on the visible answer.
    max_answer_tokens: int = 32
    # Use letter chunks as well as words when searching for similar examples.
    char_features: bool = False
