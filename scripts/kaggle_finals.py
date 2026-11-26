"""Bound a Kaggle finals session and carry validated work between notebook versions."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/kaggle/working/cairn-finals")
MARKER = "e004-files.sha256"
# Only the task 5 inference policy changed from these two published finals snapshots.
PREVIOUS_FINALS = {
    "1f16a09b37f2b8ee995f4146e4c78c973928a3e96074c43d8a62da88482826ea",
    "c62f34ccf49388aa351918ba538b2af13c8c2714f0e7afc21a7cd91504846705",
}
E003 = {1: 60, 2: 41, 3: 60, 4: 60, 5: 51, 6: 58, 7: 10}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, content):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def restore(out, inputs, current):
    def compatible(folder):
        marker = folder / MARKER
        if not marker.is_file():
            return False
        data = marker.read_bytes()
        return data == current or hashlib.sha256(data).hexdigest() in PREVIOUS_FINALS

    if out.exists() and any(out.iterdir()):
        if not compatible(out):
            raise ValueError(f"Cannot resume unidentified checkpoint at {out}")
    else:
        mounted = [p.parent for p in inputs.rglob(MARKER) if p.parent.name == "cairn-finals"]
        candidates = sorted(folder for folder in mounted if compatible(folder))
        if mounted and not candidates:
            raise ValueError(
                f"Found finals outputs with incompatible snapshot identities: {mounted}"
            )
        if len(candidates) > 1:
            raise ValueError(f"Attach only the most recent finals output. Found {candidates}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if candidates:
            print(f"Restoring {candidates[0]}")
            with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
                staged = Path(scratch) / out.name
                shutil.copytree(candidates[0], staged)
                staged.replace(out)
        else:
            out.mkdir(exist_ok=True)
            print("Starting a new finals run. No compatible saved finals output was found.")
    marker = out / MARKER
    if marker.is_file() and marker.read_bytes() != current:
        # Retain raw model replies by request hash. Old predictions and any final
        # artifacts based on them must no longer count as finished under the new policy.
        prior = out / "prior-task5-policy"
        task = out / "task5"
        affected = [
            *task.glob("*.journal*"),
            *task.glob("*.report.json"),
            task / "openseek-5.jsonl",
            out / "submission.zip",
            out / "task8/replay.json",
        ]
        for source in affected:
            if source.is_file():
                dest = prior / source.relative_to(out)
                dest.parent.mkdir(parents=True, exist_ok=True)
                source.rename(dest)
        print("Updated task 5 policy. Reusing matched model replies and other tasks' progress.")
    tmp = out / "snapshot.tmp"
    tmp.write_bytes(current)
    tmp.replace(out / MARKER)


def progress(out):
    status = {"dev": {}, "test_journal_ids": {}}
    for task in range(1, 9):
        folder = out / f"task{task}"
        journal = folder / "test.journal.jsonl"
        if journal.exists():
            ids = set()
            with journal.open(encoding="utf-8", errors="replace") as lines:
                for line in lines:
                    try:
                        ids.add(json.loads(line)["query_id"])
                    except (ValueError, KeyError, TypeError):
                        continue
            status["test_journal_ids"][task] = len(ids)
        report = folder / "dev.report.json"
        if task in E003 and report.exists():
            try:
                score = read(report)["eval"]
                if score["n"] == 60:
                    status["dev"][task] = {
                        "correct": score["correct"],
                        "n": 60,
                        "change_from_e003": score["correct"] - E003[task],
                    }
            except (ValueError, KeyError, TypeError):
                continue  # worker can be in the middle of writing the report
    save(out / "progress.json", status)
    counts = status["test_journal_ids"]
    print(f"Saved test IDs: {sum(counts.values())}/3666 {counts}")
    if status["dev"]:
        print(
            "Dev accuracy (change in correct answers vs E003):",
            ", ".join(
                f"T{t} {s['correct']}/60 ({s['change_from_e003']:+d})"
                for t, s in status["dev"].items()
            ),
        )


def run_session():
    minutes = int(os.environ.get("CAIRN_SESSION_MINUTES", "180"))
    if minutes < 1:
        raise ValueError("CAIRN_SESSION_MINUTES must be positive")
    deadline = time.monotonic() + 60 * minutes
    restore(OUT, Path("/kaggle/input"), (ROOT / MARKER).read_bytes())
    progress(OUT)
    if time.monotonic() >= deadline:
        print("Restore used this session's budget. Checkpoint ready.")
        return
    process = subprocess.Popen(
        ["bash", str(ROOT / "scripts/kaggle_e004.sh"), "finals-worker"],
        start_new_session=True,
    )
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("Session budget reached. Saving progress.")
                break
            try:
                code = process.wait(timeout=min(300, remaining))
            except subprocess.TimeoutExpired:
                progress(OUT)
                continue
            if code:
                raise SystemExit(code)
            break
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=50)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        progress(OUT)
    if not (OUT / "task8/replay.json").exists():
        print(
            "CHECKPOINT READY. Save notebook output. In the next fresh session attach "
            "that saved output as an input and run the same cell."
        )


def plan(out):
    from cairn.data import load_task
    from cairn.submit import validate_task_file
    from scripts.compare_e003 import dev_hits, dev_queries

    # Expose measured dev scores early. Keep the full target batch during preparation.
    order = [1, 3, 4, 5, 6, 7, 2]
    for split, tasks in (("dev", order), ("test", [*order, 8])):
        for task in tasks:
            folder = out / f"task{task}"
            try:
                if split == "dev":
                    dev_hits(out, task, dev_queries(task))
                    continue
                data = load_task(task)
                report = read(folder / "test.report.json")
                if (report["model"], report["split"], report["n"]) == (
                    "Qwen3-4B",
                    "test",
                    len(data.test),
                ) and not validate_task_file(folder / f"openseek-{task}.jsonl", data):
                    continue
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                pass  # missing or interrupted output: let Cairn resume its journal
            yield task, split


def finish(out):
    import torch

    from cairn.data import load_task
    from cairn.eval.replay import replay_kernels
    from cairn.submit import build_zip
    from cairn.tasks.kernel_exec import code_quality
    from cairn.types import Prediction
    from scripts.compare_e003 import dev_hits, dev_queries

    correct = sum(sum(dev_hits(out, t, dev_queries(t))) for t in E003)
    print(f"Verified dev accuracy: {correct}/420 ({correct / 420:.2%}); E003 scored 340/420.")

    archive = out / "submission.zip"
    build_zip(out, out / "submission.tmp.zip").replace(archive)
    print(f"Validated all eight test files: {archive}")
    if not torch.cuda.is_available() or os.environ.get("CAIRN_TRITON_INTERPRET") != "0":
        raise RuntimeError("Kernel replay requires the real GPU executor")
    data = load_task(8)
    folder = out / "task8"
    checkpoints = folder / "replay-cases"
    checkpoints.mkdir(exist_ok=True)
    predictions = []
    with tempfile.TemporaryDirectory() as temp:
        part = Path(temp) / "kernel.jsonl"
        for line in (folder / "openseek-8.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            ident = row["test_sample_id"]
            record = checkpoints / (hashlib.sha256(line.encode()).hexdigest() + ".json")
            if record.is_file():
                meta = read(record)
            else:
                part.write_text(line + "\n", encoding="utf-8")
                meta = replay_kernels(part, data, execute=True)["cases"][ident]
                if meta["exec_status"] == "unavailable":
                    raise RuntimeError(f"Kernel executor unavailable: {meta.get('detail')}")
                save(record, meta)
            predictions.append(Prediction(ident, row["prediction"], meta=meta))
    summary = code_quality(predictions)
    save(
        folder / "replay.json",
        {
            "execution_requested": True,
            "total_test_inputs": len(data.test),
            "summary": summary,
            "cases": {p.query_id: p.meta for p in predictions},
        },
    )
    print("Task 8 local GPU replay:", summary)
    print("FINALS COMPLETE:", archive)


if __name__ == "__main__":
    if sys.argv[1:] == ["plan"]:
        for item in plan(OUT):
            print(*item)
    elif sys.argv[1:] == ["finish"]:
        finish(OUT)
    elif not sys.argv[1:]:
        run_session()
    else:
        raise SystemExit("Expected no argument, plan or finish")
