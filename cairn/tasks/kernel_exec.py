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
from dataclasses import asdict, dataclass
from pathlib import Path

from cairn.tasks.kernel import WrapperSignature

_DRIVER = Path(__file__).with_name("_kernel_driver.py")

# Statuses that prove something, as opposed to "we could not find out".
PROVEN_GOOD = {"exec_ok"}
PROVEN_BAD = {"wrong_result", "call_error", "import_error"}
CALLABLE = {"exec_ok", "wrong_result", "called_no_reference"}


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
        "code": code,
        "name": signature.py_name,
        "full_name": signature.full_name,
        "params": [asdict(p) for p in signature.params],
        "interpret": interpret,
    }
    try:
        done = subprocess.run(
            [python_exe, str(_DRIVER)],
            input=json.dumps(job),
            capture_output=True,
            text=True,
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
