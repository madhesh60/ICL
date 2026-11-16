"""Guideline induction: the model writes down the unwritten rules of a task.

A task description says "count the nouns".  It does not say whether "train" in
"train tracks" counts, or whether "boiling" is a verb.  The labelled examples
know.  Hundreds of them are too many to keep in mind at once, even for a
person, so we do what a person would do:

    map     read the examples in small piles and note the rules each pile shows
    reduce  merge the notes into one short rulebook

The rulebook goes to the very top of every prompt, where a model pays the most
attention.  It is only kept if a with/without comparison on held-out examples
says it does not hurt (see `TaskRunner`): a confident but wrong rule is worse
than no rule.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from cairn.decoding.parse import strip_think
from cairn.llm.base import GenRequest, LLMBackend, LLMError

log = logging.getLogger(__name__)

_MAP_ASK = (
    "Above are labelled examples of the task. Write at most 6 short rules that an annotator "
    "must follow to reproduce these labels. Focus on the non-obvious decisions and edge cases "
    "the examples reveal. Do not repeat the task description. One rule per line, starting with '- '."
)
_REDUCE_ASK = (
    "Above are rule lists written from different batches of examples for the same task. Merge "
    "them into one list of at most {max_rules} short rules. Drop duplicates, drop rules that "
    "contradict each other, keep the most specific ones. One rule per line, starting with '- '."
)


def _rules_only(text: str, max_rules: int) -> str:
    lines = [line.strip() for line in strip_think(text).splitlines()]
    rules = [line for line in lines if line.startswith(("- ", "* "))]
    return "\n".join("- " + rule[2:].strip() for rule in rules[:max_rules])


class GuidelineInducer:
    def __init__(
        self,
        backend: LLMBackend,
        *,
        chunk_size: int = 60,
        n_chunks: int = 4,
        max_rules: int = 10,
        max_tokens: int = 512,
        thinking: bool = False,
    ) -> None:
        self.backend = backend
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.max_rules = max_rules
        self.max_tokens = max_tokens
        self.thinking = thinking

    def _ask(self, system: str, user: str, tag: str) -> str:
        request = GenRequest(
            messages=(("system", system), ("user", user)),
            max_tokens=self.max_tokens,
            temperature=0.0,
            thinking=self.thinking,
            tag=tag,
        )
        return self.backend.generate(request).texts[0]

    def induce(self, definition: str, rendered_examples: Sequence[str]) -> str:
        """Return a rulebook as text, or "" if the model produced nothing usable."""
        system = f"You are writing annotation guidelines.\n\n# Task\n{definition.strip()}"
        notes: list[str] = []
        for c in range(self.n_chunks):
            chunk = rendered_examples[c * self.chunk_size : (c + 1) * self.chunk_size]
            if not chunk:
                break
            try:
                rules = _rules_only(self._ask(system, "".join(chunk) + _MAP_ASK, "induce_rules_map"), 6)
            except LLMError as err:
                log.warning("guideline induction: a batch failed (%s)", err)
                continue
            if rules:
                notes.append(rules)
        if not notes:
            return ""
        if len(notes) == 1:
            return notes[0]
        merged_input = "\n\n".join(f"Batch {i + 1}:\n{n}" for i, n in enumerate(notes))
        try:
            merged = self._ask(
                system,
                f"{merged_input}\n\n{_REDUCE_ASK.format(max_rules=self.max_rules)}",
                "induce_rules_reduce",
            )
        except LLMError as err:
            log.warning("guideline induction: merge failed (%s); using the first batch", err)
            return notes[0]
        return _rules_only(merged, self.max_rules) or notes[0]
