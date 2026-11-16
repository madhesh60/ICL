from itertools import pairwise

import pytest

from cairn.induction import (
    GuidelineInducer,
    ProgramInducer,
    SandboxLimits,
    check_source,
    run_function,
)
from cairn.llm import FunctionBackend, ScriptedBackend
from cairn.types import Example

RIGHT = "import ast\ndef solve(text):\n    xs = sorted(ast.literal_eval(text))\n    return str(min(b - a for a, b in zip(xs, xs[1:])))\n"
WRONG = "import ast\ndef solve(text):\n    xs = ast.literal_eval(text)\n    return str(abs(xs[0] - xs[1]))\n"


def _gap_examples(n: int, start: int = 0) -> list[Example]:
    out = []
    for i in range(start, start + n):
        xs = [(i * 7) % 50, (i * 13) % 50 + 3, (i * 29) % 50 - 20, i % 11]
        s = sorted(xs)
        out.append(Example(f"e{i}", str(xs), str(min(b - a for a, b in pairwise(s)))))
    return out


def test_check_source_rejects_dangerous_code():
    problems = check_source("import os\ndef solve(t):\n    return open(t).read()")
    assert any("os" in p for p in problems) and any("open" in p for p in problems)
    assert any("__class__" in p for p in check_source("def solve(t):\n    return t.__class__"))
    assert check_source("def other(t): pass")  # missing solve
    assert check_source("def solve(t) return")[0].startswith("syntax error")
    assert check_source(RIGHT) == []


def test_sandbox_contains_loops_memory_and_crashes():
    limits = SandboxLimits(per_call_s=0.3, wall_s=10)
    loop = run_function("def solve(t):\n    while True:\n        pass", ["x", "y"], limits=limits)
    assert all(not r.ok and "Timeout" in r.error for r in loop)
    assert "MemoryError" in run_function("def solve(t):\n    return 'a' * 10**10", ["x"])[0].error
    crashed = run_function("def solve(t):\n    return 1 / 0", ["x"])
    assert "ZeroDivisionError" in crashed[0].error
    assert run_function(RIGHT, ["[1, 5, 9]", "[3, 3]"])[0].value == "4"


def test_sandbox_cannot_import_third_party_or_reach_files():
    rejected = run_function("import numpy\ndef solve(t):\n    return t", ["x"])
    assert rejected[0].error.startswith("rejected")


def test_program_is_repaired_from_feedback_then_accepted():
    backend = ScriptedBackend(
        [[f"```python\n{WRONG}```", "no code"], [f"<think>fix</think>\n```python\n{RIGHT}```"] * 2]
    )
    result = ProgramInducer(backend, candidates_per_round=2).induce(
        "gap", "examples", _gap_examples(30), _gap_examples(30, 100)
    )
    assert result.accepted
    assert result.program.confirm_accuracy == 1.0
    feedback = backend.requests[1].messages[-1][1]
    assert "Expected:" in feedback and "Got:" in feedback  # failures were shown back


def test_memorising_program_is_caught_by_the_hidden_confirm_set():
    verify = _gap_examples(20)
    table = {e.input: e.output for e in verify}
    cheat = f"TABLE = {table!r}\ndef solve(text):\n    return TABLE.get(text, '0')\n"
    backend = FunctionBackend(lambda r: f"```python\n{cheat}```")
    result = ProgramInducer(backend, max_rounds=2, candidates_per_round=1).induce(
        "gap", "ex", verify, _gap_examples(20, 500)
    )
    assert not result.accepted
    assert any("hidden cases" in " ".join(a.problems) for a in result.attempts)


def test_program_induction_requires_held_out_examples():
    with pytest.raises(ValueError):
        ProgramInducer(ScriptedBackend([])).induce("d", "e", [], _gap_examples(3))


def test_guideline_inducer_maps_then_reduces():
    def reply(request):
        if request.tag == "induce_rules_map":
            return "Notes:\n- compound nouns count once\n- gerunds are verbs\nsome chatter"
        return "- compound nouns count once\n- gerunds are verbs"

    backend = FunctionBackend(reply)
    rules = GuidelineInducer(backend, chunk_size=2, n_chunks=3).induce("count nouns", ["a\n\n"] * 6)
    assert rules == "- compound nouns count once\n- gerunds are verbs"
    assert backend.calls == 4  # three batches + one merge
    assert GuidelineInducer(FunctionBackend(lambda r: "nothing useful")).induce("d", ["x"]) == ""


def test_cpu_limit_is_reported_clearly():
    limits = SandboxLimits(cpu_s=1, per_call_s=60, wall_s=20)
    result = run_function(
        "def solve(t):\n    n = 0\n    while True:\n        n += 1", ["a"], limits=limits
    )
    assert "CPU time limit" in result[0].error
