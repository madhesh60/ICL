"""Recheck a saved task-8 submission part without generating new answers."""

from __future__ import annotations

import json
from pathlib import Path

from cairn.data import TaskData
from cairn.tasks.kernel import parse_wrapper_signature, static_check
from cairn.tasks.kernel_exec import code_quality, execute_kernel
from cairn.types import Prediction


def replay_kernels(path: Path, task: TaskData, *, execute: bool = False) -> dict:
    inputs = {query.id: query.input for query in task.test}
    candidates: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            sample_id, code = row["test_sample_id"], row["prediction"]
            if not isinstance(sample_id, str) or sample_id not in inputs or sample_id in candidates:
                raise ValueError("unknown or duplicate test_sample_id")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("expected a nonempty prediction string")
            candidates[sample_id] = code
        except (ValueError, KeyError, TypeError) as err:
            raise ValueError(f"{path}, line {number}: {err}") from err
    if not candidates:
        raise ValueError(f"{path} contains no predictions")

    predictions = []
    for sample_id, code in candidates.items():
        signature = parse_wrapper_signature(inputs[sample_id])
        if signature is None:
            raise ValueError(f"missing wrapper signature for {sample_id}")
        check = static_check(code, signature)
        meta = {
            "wrapper": signature.full_name,
            "static_ok": check.ok,
            "problems": check.problems,
            "exec_status": "not_run",
        }
        if execute and check.ok:
            result = execute_kernel(code, signature)
            meta.update(exec_status=result.status, detail=result.detail, inputs=result.inputs)
        predictions.append(Prediction(sample_id, code, meta=meta))

    summary = code_quality(predictions)
    if not execute:
        for key in ("callable", "matches_torch", "consistency"):
            summary[key] = None
    return {
        "execution_requested": execute,
        "total_test_inputs": len(inputs),
        "summary": summary,
        "cases": {p.query_id: p.meta for p in predictions},
    }
