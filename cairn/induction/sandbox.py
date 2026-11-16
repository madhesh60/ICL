"""Running code that a language model wrote, without letting it hurt anything.

Model-written code can loop forever, eat all the memory, or (rarely, but once
is enough) try to delete files.  We use three layers, like a seatbelt, an
airbag and a speed limit:

    1. Read before running.  The code is parsed into a tree and rejected if it
       imports anything outside a short list of harmless modules or touches
       names such as `open`, `eval`, `__import__`.
    2. Run in a separate process, never inside our own.  If it crashes or
       hangs, we kill that process and ours carries on.
    3. Hard limits set by the operating system: CPU seconds, memory, file
       size, plus a wall-clock timeout and a per-call timeout.  The child sets
       them on itself before it loads the candidate code.

This is solid protection against accidents and lazy attacks.  It is not a
vault: for code from strangers at scale, add a container or a jail (Docker,
gVisor, nsjail) around it.
"""

from __future__ import annotations

import ast
import contextlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ALLOWED_IMPORTS = frozenset(
    {
        "ast", "bisect", "collections", "decimal", "fractions", "functools", "heapq",
        "itertools", "json", "math", "operator", "re", "statistics", "string", "typing",
    }
)  # fmt: skip
FORBIDDEN_NAMES = frozenset(
    {
        "__import__", "breakpoint", "compile", "delattr", "eval", "exec", "exit", "getattr",
        "globals", "input", "locals", "open", "quit", "setattr", "vars",
    }
)  # fmt: skip

_DRIVER = r"""
import io, json, signal, sys

def _timeout(signum, frame):
    raise TimeoutError("call took too long")

job = json.loads(sys.stdin.read())
try:  # operating-system limits, set before any candidate code exists (POSIX only)
    import resource
    memory = job["memory_mb"] * 1024 * 1024
    # Soft limit first (sends SIGXCPU, a clear "CPU limit" signal), hard limit 1 s later (SIGKILL).
    resource.setrlimit(resource.RLIMIT_CPU, (job["cpu_s"], job["cpu_s"] + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass  # Windows has no `resource`; the wall-clock and per-call timeouts still apply
real_stdout = sys.stdout
sys.stdout = io.StringIO()          # swallow prints from the tested code
results = []
try:
    namespace = {"__name__": "candidate"}
    exec(compile(job["code"], "<candidate>", "exec"), namespace)
    func = namespace[job["func"]]
except BaseException as err:
    results = [{"ok": False, "error": f"load failed: {type(err).__name__}: {err}"[:300]}] * len(job["inputs"])
else:
    signal.signal(signal.SIGALRM, _timeout)
    for item in job["inputs"]:
        signal.setitimer(signal.ITIMER_REAL, job["per_call_s"])
        try:
            value = func(item)
            results.append({"ok": True, "value": value if isinstance(value, str) else repr(value)})
        except BaseException as err:
            results.append({"ok": False, "error": f"{type(err).__name__}: {err}"[:300]})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
real_stdout.write(json.dumps(results))
"""


@dataclass(frozen=True)
class SandboxLimits:
    wall_s: float = 20.0
    cpu_s: int = 10
    memory_mb: int = 512
    per_call_s: float = 1.0


@dataclass(frozen=True)
class CallResult:
    ok: bool
    value: str = ""
    error: str = ""


def check_source(code: str, required_function: str = "solve") -> list[str]:
    """Read the code without running it.  Returns a list of problems (empty = fine)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        return [f"syntax error on line {err.lineno}: {err.msg}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"access to '{node.attr}' is not allowed")
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if required_function not in defined:
        problems.append(f"the code must define a top-level function named '{required_function}'")
    return sorted(set(problems))


def run_function(
    code: str,
    inputs: Sequence[str],
    *,
    function: str = "solve",
    limits: SandboxLimits | None = None,
) -> list[CallResult]:
    """Call `function(text)` for every input inside a locked-down child process.

    Always returns one result per input.  Rejected code, crashes and timeouts
    come back as failed results with a readable reason, never as an exception.
    """
    limits = limits or SandboxLimits()
    problems = check_source(code, function)
    if problems:
        return [CallResult(ok=False, error="rejected: " + "; ".join(problems))] * len(inputs)
    if not inputs:
        return []
    job = json.dumps(
        {
            "code": code,
            "func": function,
            "inputs": list(inputs),
            "per_call_s": limits.per_call_s,
            "cpu_s": limits.cpu_s,
            "memory_mb": limits.memory_mb,
        }
    )
    with tempfile.TemporaryDirectory(prefix="cairn-sandbox-") as workdir:
        driver = Path(workdir) / "driver.py"
        driver.write_text(_DRIVER, encoding="utf-8")
        try:
            done = subprocess.run(
                # -I: ignore environment variables and the user's packages.
                # -S: do not load site-packages, so only the standard library exists.
                [sys.executable, "-I", "-S", str(driver)],
                input=job,
                capture_output=True,
                text=True,
                timeout=limits.wall_s,
                cwd=workdir,
                env={},
                # No preexec_fn: it is unsafe in programs that use threads (Cairn does).
                # The driver applies the limits itself as its first step instead.
            )
        except subprocess.TimeoutExpired:
            return [CallResult(ok=False, error="timeout: the program ran too long")] * len(inputs)
    with contextlib.suppress(json.JSONDecodeError, TypeError, KeyError):
        rows = json.loads(done.stdout)
        if len(rows) == len(inputs):
            return [
                CallResult(ok=bool(r["ok"]), value=r.get("value", ""), error=r.get("error", ""))
                for r in rows
            ]
    if done.returncode < 0:
        # Killed by a signal: SIGXCPU (24) is the CPU-seconds limit, SIGKILL (9) usually memory.
        signal_no = -done.returncode
        why = {24: "CPU time limit", 9: "killed (often the memory limit)"}.get(
            signal_no, "a signal"
        )
        return [CallResult(ok=False, error=f"stopped by the operating system: {why}")] * len(inputs)
    lines = (done.stderr or "").strip().splitlines()
    reason = lines[-1][:300] if lines else f"no output (exit code {done.returncode})"
    return [CallResult(ok=False, error=f"crashed: {reason}")] * len(inputs)
