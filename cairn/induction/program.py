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
    "Write a Python function `solve(text: str) -> str` that makes every assert above true.\n"
    "- `text` is always a Python str. If it looks like a list or a number, it is still text:\n"
    "  the brackets, quotes and commas are characters of the string. Parse it first if you\n"
    "  need its parts (ast.literal_eval does this safely).\n"
    "- Return a str that equals the right-hand side of the assert exactly.\n"
    "- Use only the Python standard library. Do not use eval or exec.\n"
    "Reply with one Python code block and nothing else."
)


@dataclass
class Attempt:
    round: int
    code: str
    verify_accuracy: float
    problems: list[str] = field(default_factory=list)
    # Why the reply ended: "stop" is normal, "length" means the token budget ran out
    # (usually while still thinking), so no code could follow.
    finish: str = ""
    thinking: bool = True
    # True when this exact code had already been tried and failed.
    repeated: bool = False


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


def _diagnose(result: CallResult, example: Example) -> str:
    """One counter-example, written so that the KIND of mistake is visible.

    E001 lesson: "Got: ['cutting', 'y', ...]" looked like a list to the model, which then
    rewrote the same function four times.  Shown with repr() and a plain-words note
    ("returned its input unchanged"), the mistake names itself.
    """
    lines = [f"Call:     solve({example.input!r})", f"Expected: {example.output!r}"]
    if not result.ok:
        lines.append(f"Got:      ERROR {result.error}")
        return "\n".join(lines)
    lines.append(f"Got:      {result.value!r}")
    notes = []
    if result.kind != "str":
        notes.append(f"it returned a {result.kind}, not a str")
    if result.value == example.input:
        notes.append("it returned its input unchanged")
    if notes:
        lines.append("Note:     " + "; ".join(notes))
    return "\n".join(lines)


def _score(results: Sequence[CallResult], examples: Sequence[Example]) -> tuple[float, list[str]]:
    """Accuracy and a few readable counter-examples."""
    wrong: list[str] = []
    hits = 0
    for result, example in zip(results, examples, strict=True):
        if result.ok and result.kind == "str" and result.value.strip() == example.output.strip():
            hits += 1
        elif len(wrong) < 5:
            wrong.append(_diagnose(result, example))
    return hits / max(len(examples), 1), wrong


def _normalise_code(code: str) -> str:
    """Code with comments and blank lines removed, to recognise a repeated attempt."""
    kept = []
    for line in code.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if stripped.strip():
            kept.append(stripped)
    return "\n".join(kept)


class ProgramInducer:
    def __init__(
        self,
        backend: LLMBackend,
        *,
        max_rounds: int = 4,
        candidates_per_round: int = 2,
        accept_accuracy: float = 1.0,
        max_tokens: int = 4096,
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
        thinking = self.thinking
        # Failed programs seen so far, by normalised source: running the same code twice
        # teaches nothing, and telling the model it repeated itself does.
        failed: dict[str, Attempt] = {}
        for round_index in range(self.max_rounds):
            request = GenRequest(
                messages=tuple(base) + feedback,
                max_tokens=self.max_tokens,
                # Round 0 with one candidate is greedy; otherwise sample for variety.
                temperature=0.0 if self.candidates_per_round == 1 and round_index == 0 else 0.7,
                top_p=0.95,
                n=self.candidates_per_round,
                seed=self.seed + round_index,
                thinking=thinking,
                tag="induce_program",
            )
            try:
                response = self.backend.generate(request)
            except LLMError as err:
                log.warning("program induction stopped: %s", err)
                break
            replies = response.texts
            finishes = list(response.finish_reasons) + [""] * len(replies)

            best: Attempt | None = None
            for reply, finish in zip(replies, finishes, strict=False):
                code = extract_code(reply)
                if not code:
                    why = (
                        "token budget ran out before any code"
                        if finish == "length"
                        else "no code block found"
                    )
                    attempts.append(Attempt(round_index, reply[:200], 0.0, [why], finish, thinking))
                    log.info("program round %d: %s", round_index, why)
                    continue
                seen = failed.get(_normalise_code(code))
                if seen is not None:
                    accuracy, wrong = seen.verify_accuracy, list(seen.problems)
                else:
                    accuracy, wrong = _score(
                        run_function(code, [e.input for e in verify], limits=self.limits), verify
                    )
                attempt = Attempt(round_index, code, accuracy, wrong, finish, thinking)
                attempt.repeated = seen is not None
                attempts.append(attempt)
                log.info(
                    "program round %d: %.1f%% right on %d verify examples%s",
                    round_index,
                    100 * accuracy,
                    len(verify),
                    f"; first failure: {wrong[0][:160]!r}" if wrong else "",
                )
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
                failed.setdefault(_normalise_code(code), attempt)
                if best is None or attempt.verify_accuracy > best.verify_accuracy:
                    best = attempt

            if best is None:
                # Not one reply contained code.  With thinking on, the usual reason is
                # that the whole budget went into the thinking block.  Ask again without
                # thinking: a five-line function does not need a long deliberation, and
                # a plain reply cannot be swallowed by it.
                if thinking:
                    log.info(
                        "program round %d gave no code; retrying without thinking", round_index
                    )
                    thinking = False
                    feedback = ()
                else:
                    feedback = (("user", "No Python code block was found. " + _ASK),)
                continue
            listing = "\n\n".join(best.problems) or "It fails on hidden cases."
            repeat_note = (
                "You wrote this same function before, and it failed the same way. Do not "
                "submit it again: change the approach.\n\n"
                if getattr(best, "repeated", False)
                else ""
            )
            feedback = (
                ("assistant", f"```python\n{best.code}\n```"),
                (
                    "user",
                    f"{repeat_note}That function is right on {best.verify_accuracy:.0%} of the "
                    f"test cases. Failures:\n\n{listing}\n\nRemember that `text` is a str. "
                    "Find the general rule and rewrite `solve`. Reply with one Python code block.",
                ),
            )
        return InductionResult(None, attempts)
