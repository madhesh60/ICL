"""The eight task cards.

Every regular expression here was checked against every example and every test
input in the data files (zero misses).  If a pattern ever fails to match, the
view falls back to the raw input, so retrieval degrades gently instead of crashing.
"""

from __future__ import annotations

import re
from dataclasses import replace

from cairn.decoding.constraints import Choice, Regex
from cairn.decoding.parse import (
    extract_code,
    make_label_normalizer,
    norm_int_list,
    norm_integer,
    norm_text,
    norm_trivia,
)
from cairn.tasks.base import TaskSpec, View
from cairn.tasks.signals import sentence_pair_overlap

_T2 = re.compile(r"^Sentence: '(.*)'\. Count the number of (nouns|verbs) in this sentence\.$", re.S)
_T6 = re.compile(r"^Sentence 1: (.*?)\s*Sentence 2: (.*?)\s*Genre: (.*?)\.?$", re.S)
_T7 = re.compile(r"^Category: (.*?)\s*\nClue: (.*)$", re.S)
_T8_PREAMBLE = re.compile(r"^You are an? expert in .*?\n", re.S)
_T8_TRAILER = "After generation, verify if the Triton wrapper aligns"


def _plain_view(text: str) -> View:
    return View(text=text)


def _list_view(text: str) -> View:
    # Lists of the same length are solved the same way, so group by length.
    return View(text=text, stratum=str(text.count(",") + 1))


def _t2_view(text: str) -> View:
    match = _T2.match(text)
    if not match:
        return View(text=text)
    return View(text=match.group(1), stratum=match.group(2))


def _t6_view(text: str) -> View:
    match = _T6.match(text)
    if not match:
        return View(text=text)
    return View(text=f"{match.group(1)} {match.group(2)}", stratum=match.group(3).strip())


def _t7_view(text: str) -> View:
    match = _T7.match(text)
    if not match:
        return View(text=text)
    return View(text=match.group(2), boosted=match.group(1))


def _t8_view(text: str) -> View:
    body = _T8_PREAMBLE.sub("", text, count=1)
    return View(text=body.split(_T8_TRAILER, 1)[0].strip())


TASKS: dict[int, TaskSpec] = {
    1: TaskSpec(
        task_id=1,
        name="closest_integers",
        kind="deterministic",
        min_context_tokens=30_000,
        view=_list_view,
        normalize=norm_integer,
        constraint=Regex(r"(0|[1-9][0-9]{0,3})"),
        answer_rule="Reply with a single non-negative integer and nothing else.",
        max_answer_tokens=8,
    ),
    2: TaskSpec(
        task_id=2,
        name="count_nouns_verbs",
        kind="classification",
        min_context_tokens=30_000,
        view=_t2_view,
        normalize=norm_integer,
        answer_rule="Reply with a single digit and nothing else.",
        max_answer_tokens=4,
    ),
    3: TaskSpec(
        task_id=3,
        name="collatz_conjecture",
        kind="deterministic",
        min_context_tokens=30_000,
        view=_list_view,
        normalize=norm_int_list,
        constraint=Regex(r"\[[0-9]+(, [0-9]+)*\]"),
        answer_rule="Reply with the resulting list only, written like [1, 2, 3].",
        max_answer_tokens=64,
    ),
    4: TaskSpec(
        task_id=4,
        name="conala_concat_strings",
        kind="deterministic",
        min_context_tokens=30_000,
        view=_list_view,
        normalize=norm_text,
        answer_rule="Reply with the joined string only: no quotes, no spaces added, no explanation.",
        max_answer_tokens=96,
    ),
    5: TaskSpec(
        task_id=5,
        name="tweet_sadness_detection",
        kind="classification",
        min_context_tokens=30_000,
        view=_plain_view,
        normalize=norm_text,
        answer_rule='Reply with exactly "Sad" or "Not sad" and nothing else.',
        max_answer_tokens=6,
        char_features=True,
    ),
    6: TaskSpec(
        task_id=6,
        name="mnli_same_genre_classification",
        kind="classification",
        min_context_tokens=30_000,
        view=_t6_view,
        normalize=norm_text,
        answer_rule="Reply with exactly Y or N and nothing else.",
        max_answer_tokens=4,
        signals=sentence_pair_overlap,
    ),
    7: TaskSpec(
        task_id=7,
        name="jeopardy_answer_generation",
        kind="generation",
        min_context_tokens=30_000,
        view=_t7_view,
        normalize=norm_trivia,
        answer_rule="Reply with the answer only, in lower case, as short as the examples.",
        max_answer_tokens=24,
        char_features=True,
    ),
    8: TaskSpec(
        task_id=8,
        name="kernel_generation",
        kind="code",
        min_context_tokens=16_000,
        view=_t8_view,
        normalize=extract_code,
        answer_rule=(
            "Reply with one Python code block that contains the Triton kernel(s) and the wrapper "
            "function. The wrapper must have exactly the name and parameters given."
        ),
        output_style="code",
        max_answer_tokens=3072,
    ),
}


def get_task(task_id: int, labels: tuple[str, ...] | None = None) -> TaskSpec:
    """Return the card for a task.

    For classification tasks pass the labels seen in the data; the card is then
    completed with a closed label set, a matching constraint and a normalizer.
    The label set comes from the data, not from this file, so it cannot go stale.
    """
    spec = TASKS[task_id]
    if spec.kind == "classification" and labels:
        ordered = tuple(sorted(set(labels)))
        spec = replace(
            spec,
            labels=ordered,
            constraint=Choice(ordered),
            normalize=make_label_normalizer(ordered),
        )
    return spec
