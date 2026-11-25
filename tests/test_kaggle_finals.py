import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cairn.data import load_task
from cairn.pipeline.journal import Journal
from cairn.submit import write_task_file
from cairn.types import Prediction
from scripts import kaggle_finals as finals


def test_restore_then_resume_interrupted_journal(tmp_path):
    previous = tmp_path / "input" / "version1" / "cairn-finals"
    task = previous / "task2"
    task.mkdir(parents=True)
    (previous / finals.MARKER).write_bytes(b"snapshot")
    journal = task / "test.journal.jsonl"
    Journal(journal).append(Prediction("done", "3"))
    with journal.open("ab") as handle:
        handle.write(b'{"answer":"\xe2')
    out = tmp_path / "working" / "cairn-finals"
    finals.restore(out, tmp_path / "input", b"snapshot")
    restored = Journal(out / "task2" / journal.name)
    restored.append(Prediction("resumed", "4"))
    assert set(restored.load()) == {"done", "resumed"}
    assert journal.read_bytes().endswith(b'{"answer":"\xe2')
    finals.restore(out, tmp_path / "input", b"snapshot")
    assert len(restored.load()) == 2
    with pytest.raises(ValueError, match="unidentified"):
        finals.restore(out, tmp_path / "input", b"incompatible")


def test_plan_skips_verified_work_and_requeues_incomplete_test(tmp_path):
    assert len(list(finals.plan(tmp_path))) == 15
    task = tmp_path / "task2"
    task.mkdir()
    source = Path(__file__).resolve().parents[1] / "dev_results_now/task2"
    for name in ("dev.journal.jsonl", "dev.report.json"):
        shutil.copyfile(source / name, task / name)
    queries = load_task(2).test
    answers = [Prediction(q.id, "fixture") for q in queries]
    target = task / "openseek-2.jsonl"
    write_task_file(target, answers)
    (task / "test.report.json").write_text(
        json.dumps({"model": "Qwen3-4B", "split": "test", "n": len(queries)})
    )
    assert all(t != 2 for t, _ in finals.plan(tmp_path))
    write_task_file(target, answers[:-1])
    assert (2, "test") in list(finals.plan(tmp_path))


def test_budget_stops_worker_and_keeps_checkpoint(tmp_path, monkeypatch):
    out = tmp_path / "cairn-finals"
    (tmp_path / finals.MARKER).write_bytes(b"snapshot")
    monkeypatch.setattr(finals, "ROOT", tmp_path)
    monkeypatch.setattr(finals, "OUT", out)
    monkeypatch.setenv("CAIRN_SESSION_MINUTES", "1")
    ticks = iter([0, 1, 2, 61])
    monkeypatch.setattr(finals.time, "monotonic", lambda: next(ticks))
    worker = Mock(pid=123)
    worker.poll.return_value = None
    worker.wait.side_effect = [subprocess.TimeoutExpired("bash", 58), -15]
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=worker))
    stop = Mock()
    monkeypatch.setattr(finals.os, "killpg", stop, raising=False)
    finals.run_session()
    stop.assert_called_once_with(123, finals.signal.SIGTERM)
    assert (out / "progress.json").is_file()
