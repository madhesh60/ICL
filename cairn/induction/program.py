"""Program induction: the model reads the examples and writes the rule as code.

Three of the tasks are exact procedures (smallest gap between numbers, one
Collatz step, gluing strings together).  A 4B model that "eyeballs" a list of
ten numbers makes slips.  The same model can usually *describe* the procedure
as five lines of Python, and Python does not make slips.

The danger is a program that is confidently wrong.  We never trust it; we test it:

    show      examples the model sees (they are in the prompt)
    verify    held-out examples used to test each candidate program; failures
              are shown back to the model so it can repair the program
    confirm   a second held-out set the repair loop never sees.  A program that
              passes `verify` only because it was patched for those exact cases
              fails here.

Only a program that reproduces every `verify` and every `confirm` answer is
accepted.  Otherwise the task falls back to ordinary in-context labelling.
This is the left-over-examples idea at its plainest: the examples that did not
fit in the prompt become a unit-test suite.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from cairn.decoding.parse import extract_code
from cairn.induction.sandbox import CallResult, SandboxLimits, run_function
from cairn.llm.base import GenRequest, LLMBackend, LLMError
from cairn.types import Example

log = logging.getLogger(__name__)

_SYSTEM = (
    "You write short, exact Python functions. You study input/output examples, work out the "
    "rule that maps each input to its output, and write that rule as code."
)
_ASK = (
    "Write a Python function `solve(text: str) -> str`.\n"
    "- `text` is exactly the string shown after 'Input:' in the examples.\n"
    "- It must return exactly the string shown after 'Output:' (same spacing and punctuation).\n"
    "- Use only the Python standard library.\n"
    "Reply with one Python code block and nothing else."
)


@dataclass
class Attempt:
    round: int
    code: str
    verify_accuracy: float
    problems: list[str] = field(default_factory=list)


@dataclass
class InducedProgram:
    code: str
    verify_accuracy: float
    confirm_accuracy: float
    limits: SandboxLimits = field(default_factory=SandboxLimits)

    def run(self, inputs: Sequence[str]) -> list[str | None]:
        """Apply the program.  `None` marks inputs where it failed (use a fallback there)."""
        results = run_function(self.code, inputs, limits=self.limits)
        return [r.value if r.ok else None for r in results]


@dataclass
class InductionResult:
    program: InducedProgram | None
    attempts: list[Attempt]

    @property
    def accepted(self) -> bool:
        return self.program is not None


def _score(results: Sequence[CallResult], examples: Sequence[Example]) -> tuple[float, list[str]]:
    """Accuracy and a few readable counter-examples."""
    wrong: list[str] = []
    hits = 0
    for result, example in zip(results, examples, strict=True):
        if result.ok and result.value.strip() == example.output.strip():
            hits += 1
        elif len(wrong) < 5:
            got = result.value if result.ok else f"ERROR {result.error}"
            wrong.append(f"Input: {example.input}\nExpected: {example.output}\nGot: {got}")
    return hits / max(len(examples), 1), wrong


class ProgramInducer:
    def __init__(
        self,
        backend: LLMBackend,
        *,
        max_rounds: int = 4,
        candidates_per_round: int = 2,
        accept_accuracy: float = 1.0,
        max_tokens: int = 2048,
        thinking: bool = True,
        limits: SandboxLimits | None = None,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.max_rounds = max_rounds
        self.candidates_per_round = candidates_per_round
        self.accept_accuracy = accept_accuracy
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.limits = limits or SandboxLimits()
        self.seed = seed

    def induce(
        self,
        definition: str,
        examples_text: str,
        verify: Sequence[Example],
        confirm: Sequence[Example],
    ) -> InductionResult:
        """Propose programs, test them, feed failures back, accept only a proven one.

        `examples_text` is the block of rendered examples (the same long context
        the annotation prompts use).
        """
        if not verify or not confirm:
            raise ValueError("program induction needs held-out examples to verify and to confirm")
        base = [
            ("system", _SYSTEM),
            ("user", f"# Task\n{definition.strip()}\n\n{examples_text}\n{_ASK}"),
        ]
        attempts: list[Attempt] = []
        feedback: tuple[tuple[str, str], ...] = ()
        for round_index in range(self.max_rounds):
            request = GenRequest(
                messages=tuple(base) + feedback,
                max_tokens=self.max_tokens,
                # Round 0 with one candidate is greedy; otherwise sample for variety.
                temperature=0.0 if self.candidates_per_round == 1 and round_index == 0 else 0.7,
                top_p=0.95,
                n=self.candidates_per_round,
                seed=self.seed + round_index,
                thinking=self.thinking,
                tag="induce_program",
            )
            try:
                replies = self.backend.generate(request).texts
            except LLMError as err:
                log.warning("program induction stopped: %s", err)
                break

            best: Attempt | None = None
            for reply in replies:
                code = extract_code(reply)
                if not code:
                    attempts.append(Attempt(round_index, reply[:200], 0.0, ["no code block found"]))
                    continue
                accuracy, wrong = _score(
                    run_function(code, [e.input for e in verify], limits=self.limits), verify
                )
                attempt = Attempt(round_index, code, accuracy, wrong)
                attempts.append(attempt)
                if accuracy >= self.accept_accuracy:
                    confirm_accuracy, _ = _score(
                        run_function(code, [e.input for e in confirm], limits=self.limits), confirm
                    )
                    if confirm_accuracy >= self.accept_accuracy:
                        log.info(
                            "program accepted in round %d: %d/%d verify, %d/%d confirm",
                            round_index, len(verify), len(verify), len(confirm), len(confirm),
                        )  # fmt: skip
                        return InductionResult(
                            InducedProgram(code, accuracy, confirm_accuracy, self.limits), attempts
                        )
                    # Passed the shown tests, failed the hidden ones: it memorised
                    # cases instead of finding the rule.  Say so without revealing them.
                    attempt.problems = ["correct on the shown cases but wrong on hidden cases"]
                if best is None or attempt.verify_accuracy > best.verify_accuracy:
                    best = attempt

            if best is None:
                feedback = (("user", "No Python code block was found. " + _ASK),)
                continue
            listing = "\n\n".join(best.problems) or "It fails on hidden cases."
            feedback = (
                ("assistant", f"```python\n{best.code}\n```"),
                (
                    "user",
                    f"That function is right on {best.verify_accuracy:.0%} of the test cases. "
                    f"Failures:\n\n{listing}\n\nFind the general rule and rewrite `solve`. "
                    "Reply with one Python code block.",
                ),
            )
        return InductionResult(None, attempts)
