"""Writing predictions in the official format and checking them before they leave.

Format (from the competition's outputs/README): one zip, eight files named
`openseek-<task>.jsonl`, each line `{"test_sample_id": ..., "prediction": ...}`.

A submission is checked against the data files before it is zipped: every test
id present exactly once, no empty prediction, no extra ids.  Finding a missing
line here takes a second; finding it on a leaderboard takes a wasted submission.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

from cairn.data import TaskData, load_task
from cairn.types import Prediction


class SubmissionError(ValueError):
    pass


def write_task_file(path: Path | str, predictions: Sequence[Prediction]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for p in predictions:
            row = {"test_sample_id": p.query_id, "prediction": p.answer}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Write to a temporary name, then rename: a reader never sees a half-written file.
    temporary.replace(path)
    return path


def validate_task_file(path: Path | str, task: TaskData) -> list[str]:
    """Return a list of problems (empty list = the file is a valid submission part)."""
    problems: list[str] = []
    expected = {q.id for q in task.test}
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                sample_id, prediction = row["test_sample_id"], row["prediction"]
            except (json.JSONDecodeError, KeyError, TypeError):
                problems.append(
                    f"line {number}: not a JSON object with test_sample_id and prediction"
                )
                continue
            if sample_id in seen:
                problems.append(f"line {number}: duplicate id {sample_id}")
            seen.add(sample_id)
            if sample_id not in expected:
                problems.append(f"line {number}: unknown id {sample_id}")
            if prediction is None or not str(prediction).strip():
                problems.append(f"line {number}: empty prediction for {sample_id}")
    missing = expected - seen
    if missing:
        problems.append(f"{len(missing)} test ids have no prediction (e.g. {sorted(missing)[0]})")
    return problems


def build_zip(run_dir: Path | str, out: Path | str, data_dir: Path | str | None = None) -> Path:
    """Collect the eight task files from a run folder, validate them, and zip them."""
    run_dir, out = Path(run_dir), Path(out)
    files: list[Path] = []
    problems: list[str] = []
    for task_id in range(1, 9):
        path = run_dir / f"task{task_id}" / f"openseek-{task_id}.jsonl"
        if not path.exists():
            problems.append(
                f"task {task_id}: {path} is missing (run `cairn run --task {task_id} --split test`)"
            )
            continue
        issues = validate_task_file(path, load_task(task_id, data_dir))
        problems.extend(f"task {task_id}: {issue}" for issue in issues[:5])
        files.append(path)
    if problems:
        raise SubmissionError("the submission is not complete:\n  " + "\n  ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)  # flat: no folders inside the zip
    return out
