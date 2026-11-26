"""Actually running a generated Triton kernel, in a separate process.

Reading code (the static check) catches wrong names and wrong signatures.  It
cannot tell whether `x + y` should have been `x / y`.  Only running the code
and comparing numbers can.

This module starts `_kernel_driver.py` in its own process, with a time limit,
and turns whatever happens (success, wrong numbers, exception, crash, hang)
into one small report.  On a machine without a GPU it uses Triton's interpreter
mode, so the check also works on a laptop or in CI.

Why a separate process and not a function call?  A broken kernel can crash the
whole Python interpreter (segmentation fault) or never return.  A crash in a
child process is just another result for us to read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from cairn.tasks.kernel import WrapperSignature
from cairn.types import Prediction

_DRIVER = Path(__file__).with_name("_kernel_driver.py")

# Statuses that prove something, as opposed to "we could not find out".
PROVEN_GOOD = {"exec_ok"}
PROVEN_BAD = {"wrong_result", "call_error", "import_error"}
CALLABLE = {"exec_ok", "wrong_result", "called_no_reference", "insufficient_trials"}


@dataclass(frozen=True)
class ExecReport:
    status: str
    detail: str = ""
    inputs: str = ""

    @property
    def conclusive(self) -> bool:
        return self.status in PROVEN_GOOD or self.status in PROVEN_BAD

    def feedback(self) -> str:
        """One sentence the model can act on, or "" when there is nothing to fix."""
        if self.status == "wrong_result":
            return f"The code runs but gives the wrong result: {self.detail}"
        if self.status in {"call_error", "import_error", "unconfirmed_error"}:
            return f"The code fails when executed: {self.detail}"
        if self.status == "timeout":
            return "The code did not finish in time; check for loops that never end."
        return ""


def execute_kernel(
    code: str,
    signature: WrapperSignature,
    *,
    python_exe: str | None = None,
    interpret: bool | None = None,
    timeout_s: float = 180.0,
) -> ExecReport:
    """Run the wrapper on guessed inputs and compare with PyTorch when possible.

    python_exe  interpreter that has torch and triton installed (default: this one,
                or the CAIRN_TRITON_PYTHON environment variable)
    interpret   True = CPU interpreter mode, False = real GPU, None = decide from
                CAIRN_TRITON_INTERPRET (default: interpreter mode)
    """
    python_exe = python_exe or os.environ.get("CAIRN_TRITON_PYTHON") or sys.executable
    if interpret is None:
        interpret = os.environ.get("CAIRN_TRITON_INTERPRET", "1") != "0"
    job = {
        "name": signature.py_name,
        "full_name": signature.full_name,
        "params": [asdict(p) for p in signature.params],
        "interpret": interpret,
    }
    try:
        # Triton reads @jit function source during import and again during compilation.
        # The parent owns the directory so even a crashed or timed-out child is cleaned up.
        with tempfile.TemporaryDirectory(prefix="cairn-kernel-") as directory:
            source = Path(directory) / "candidate.py"
            source.write_text(code, encoding="utf-8")
            done = subprocess.run(
                [python_exe, str(_DRIVER)],
                input=json.dumps({**job, "source": str(source.resolve())}),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired:
        return ExecReport("timeout", f"no result after {timeout_s:.0f}s")
    except OSError as err:
        return ExecReport("unavailable", f"could not start {python_exe}: {err}")
    try:
        row = json.loads(done.stdout)
        return ExecReport(row["status"], row.get("detail", ""), row.get("inputs", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        tail = (done.stderr or "").strip().splitlines()[-3:]
        return ExecReport("call_error", "the process crashed: " + " | ".join(tail)[:500])


def consistency_score(n_called: int, n_correct: int, n_total: int) -> float:
    """The official task-8 metric.

        S_call = called / total        S_exec = correct / total
        S = S_exec * (1 - (S_call - S_exec) / S_call)  =  S_exec^2 / S_call

    Worth staring at: for a fixed number of correct kernels, every kernel that
    runs but computes the wrong thing makes the score *smaller*.  The metric
    rewards code that is either right or fails openly, and punishes code that
    quietly returns wrong numbers.  That is why the repair loop works hardest on
    kernels that are callable but wrong.
    """
    if n_total <= 0 or n_called <= 0:
        return 0.0
    s_call = n_called / n_total
    s_exec = n_correct / n_total
    return s_exec * (1.0 - (s_call - s_exec) / s_call)


def code_quality(predictions: Sequence[Prediction]) -> dict[str, object]:
    """Summarise a batch of code answers from the stages they reached.

    `callable` counts kernels that ran and returned tensors (right or wrong);
    `matches_torch` counts kernels proven equal to the PyTorch function of the
    same name.  Kernels that could not be executed here count as neither, so on
    a machine without triton these two numbers are lower bounds.
    """
    n = len(predictions)
    statuses = [str(p.meta.get("exec_status", "")) or "none" for p in predictions]
    # In older journals a runtime error demoted a statically clean candidate to tier 1.
    # Reconstruct the separate stage from the fact that execution was attempted.
    static_ok = sum(
        bool(
            p.meta.get(
                "static_ok",
                int(p.meta.get("tier", 0)) >= 2 or status not in {"none", "not_run", "abstained"},
            )
        )
        for p, status in zip(predictions, statuses, strict=True)
    )
    called = sum(1 for s in statuses if s in CALLABLE)
    correct = sum(1 for s in statuses if s in PROVEN_GOOD)
    histogram: dict[str, int] = {}
    for status in statuses:
        histogram[status] = histogram.get(status, 0) + 1
    return {
        "n": n,
        "static_ok": static_ok / n if n else 0.0,
        "callable": called / n if n else 0.0,
        "matches_torch": correct / n if n else 0.0,
        "consistency": consistency_score(called, correct, n),
        # Why kernels did not count as callable: "unavailable" means the executor could
        # not run at all (no torch/triton), "unconfirmed_error" means every guessed input
        # raised.  Read this before blaming the model.
        "statuses": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
    }
