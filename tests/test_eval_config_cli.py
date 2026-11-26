import json
import zipfile

import numpy as np
import pytest

from cairn.cli import main
from cairn.config import ConfigError, load_config
from cairn.context import ContextConfig
from cairn.data import TaskData
from cairn.eval import bootstrap_ci, evaluate, mcnemar_exact, wilson_interval
from cairn.eval.autotune import successive_halving
from cairn.llm import DemoReaderBackend
from cairn.pipeline import RunConfig, StrategyConfig, TaskRunner
from cairn.submit import SubmissionError, build_zip, validate_task_file, write_task_file
from cairn.types import Prediction, Query
from tests.conftest import make_sentiment_pool

# ------------------------------------------------------------------ stats


def test_wilson_interval_is_sane_at_the_edges():
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0.2 < high < 0.35
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high and high - low < 0.21
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_only_counts_disagreements():
    same = [True, False] * 50
    assert mcnemar_exact(same, same) == (0, 0, 1.0)
    a = [True] * 30 + [False] * 70
    b = [False] * 30 + [True] * 70
    only_a, only_b, p = mcnemar_exact(a, b)
    assert (only_a, only_b) == (30, 70) and p < 0.001
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_bootstrap_ci_contains_the_mean():
    data = np.random.default_rng(0).random(200)
    low, high = bootstrap_ci(data)
    assert low < data.mean() < high


def test_evaluate_breaks_down_by_source():
    queries = [Query("a", "x", "1"), Query("b", "y", "2"), Query("c", "z", "3")]
    predictions = [
        Prediction("a", "1", 0.9, "score"),
        Prediction("b", " 2 ", 0.8, "score"),
        Prediction("c", "0", 0.0, "fallback_knn"),
    ]
    report = evaluate(predictions, queries)
    assert report.correct == 2 and report.by_source["fallback_knn"]["accuracy"] == 0.0
    with pytest.raises(ValueError):
        evaluate([Prediction("zzz", "1")], queries)


# ----------------------------------------------------------------- config


def test_config_merges_task_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "defaults:\n  workers: 3\n  context: {tail_k: 5}\ntasks:\n  7: {context: {tail_k: 20}}\n"
    )
    config = load_config(path)
    assert config.for_task(1).workers == 3 and config.for_task(1).context.tail_k == 5
    assert config.for_task(7).context.tail_k == 20 and config.for_task(7).workers == 3


@pytest.mark.parametrize(
    "text",
    [
        "defaults:\n  context: {tail_kk: 5}\n",  # typo inside a section
        "defaults:\n  wokers: 2\n",  # typo at run level
        "backend: {kind: openai, base_ur: x}\n",
        "extras: {}\n",
        "tasks:\n  9: {}\n",
    ],
)
def test_config_typos_fail_loudly(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)


def test_shipped_configs_load():
    for name in ("configs/default.yaml", "configs/offline.yaml"):
        load_config(name)


# ----------------------------------------------------------------- submit


def _fake_task(task_id: int, n: int = 3) -> TaskData:
    return TaskData(task_id, "t", "d", (), tuple(Query(f"{task_id}-{i}", "x") for i in range(n)))


def test_validate_catches_missing_duplicate_empty_and_unknown(tmp_path):
    task = _fake_task(1)
    path = write_task_file(
        tmp_path / "f.jsonl",
        [Prediction("1-0", "a"), Prediction("1-0", "a"), Prediction("1-9", "")],
    )
    problems = " ".join(validate_task_file(path, task))
    assert "duplicate" in problems and "unknown id" in problems and "empty prediction" in problems
    assert "have no prediction" in problems
    good = write_task_file(tmp_path / "g.jsonl", [Prediction(f"1-{i}", "a") for i in range(3)])
    assert validate_task_file(good, task) == []
    row = json.loads(good.read_text().splitlines()[0])
    assert set(row) == {"test_sample_id", "prediction"}


def test_build_zip_is_flat_and_refuses_incomplete_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cairn.submit.load_task", lambda task_id, data_dir=None: _fake_task(task_id)
    )
    run = tmp_path / "run"
    for task_id in range(1, 8):
        write_task_file(run / f"task{task_id}" / f"openseek-{task_id}.jsonl",
                        [Prediction(f"{task_id}-{i}", "a") for i in range(3)])  # fmt: skip
    with pytest.raises(SubmissionError, match="task 8"):
        build_zip(run, tmp_path / "s.zip")
    write_task_file(
        run / "task8" / "openseek-8.jsonl", [Prediction(f"8-{i}", "code") for i in range(3)]
    )
    out = build_zip(run, tmp_path / "s.zip")
    names = zipfile.ZipFile(out).namelist()
    assert sorted(names) == sorted(f"openseek-{i}.jsonl" for i in range(1, 9))


# --------------------------------------------------------------- autotune


def test_successive_halving_picks_the_better_setting(counter):
    pool = make_sentiment_pool(1200)
    task = TaskData(5, "toy", "sad?", tuple(pool), ())
    ctx = ContextConfig(min_demo_tokens=600, tail_k=6, tail_per_label=1)
    base = RunConfig(
        context=ctx, strategy=StrategyConfig(calibration="none", escalate=False), workers=2
    )
    candidates = {
        "good": lambda: TaskRunner(task, DemoReaderBackend(), counter, base),
        "noisy": lambda: TaskRunner(task, DemoReaderBackend(noise=0.6, seed=1), counter, base),
        "very-noisy": lambda: TaskRunner(task, DemoReaderBackend(noise=0.9, seed=2), counter, base),
    }
    dev = TaskRunner(task, DemoReaderBackend(), counter, base).splits.as_queries("dev")[:80]
    result = successive_halving(candidates, dev, start=20)
    assert result.winner == "good"
    assert len(result.trials["very-noisy"].rounds) == 1  # dropped after the first round
    assert "| good **(winner)**" in result.table()
    assert result.runner_up == "noisy" and result.compared_on == 20 and result.p_value is not None


# -------------------------------------------------------------------- cli


@pytest.mark.data
def test_cli_inspect_and_baseline(capsys):
    assert main(["inspect", "--task", "1,8"]) == 0
    assert "kernel_generation" in capsys.readouterr().out
    assert main(["baseline", "--task", "5"]) == 0
    assert "majority" in capsys.readouterr().out


@pytest.mark.data
def test_cli_offline_run_writes_journal_and_report(tmp_path, capsys):
    run_dir = tmp_path / "run"
    code = main(["--config", "configs/offline.yaml", "run", "--task", "6", "--split", "dev",
                 "--limit", "8", "--run-dir", str(run_dir)])  # fmt: skip
    assert code == 0
    report = json.loads((run_dir / "task6" / "dev.report.json").read_text())
    assert report["eval"]["n"] == 8 and report["n"] == 8
    assert len((run_dir / "task6" / "dev.journal.jsonl").read_text().splitlines()) == 8
    assert main(["submit", "--run-dir", str(run_dir), "--out", str(tmp_path / "s.zip")]) == 1


@pytest.mark.data
def test_cli_refuses_to_resume_with_different_settings(tmp_path):
    run_dir = str(tmp_path / "run")
    base = ["run", "--task", "5", "--split", "dev", "--limit", "4", "--run-dir", run_dir]
    assert main(["--config", "configs/offline.yaml", *base]) == 0
    other = tmp_path / "other.yaml"
    other.write_text("backend: {kind: demo-reader}\ndefaults: {workers: 1}\n")
    with pytest.raises(SystemExit, match="different settings"):
        main(["--config", str(other), *base])
    assert main(["--config", str(other), *base, "--fresh"]) == 0


def test_doctor_reports_each_feature_against_a_fake_server():
    from cairn.doctor import run_checks
    from cairn.llm import OpenAICompatBackend, RetryPolicy
    from tests.fake_server import FakeLLMServer

    with FakeLLMServer() as server:
        backend = OpenAICompatBackend(server.base_url, retry=RetryPolicy(max_attempts=1))
        checks = {c.name: c for c in run_checks(backend)}
    assert checks["server reachable"].ok
    assert checks["token probabilities (logprobs)"].ok
    assert checks["constrained answers (structured outputs)"].ok
    assert checks["thinking mode"].ok
    assert checks["prefix cache"].ok  # the fake server reports cached tokens


def test_doctor_explains_an_unreachable_server():
    from cairn.doctor import run_checks
    from cairn.llm import OpenAICompatBackend, RetryPolicy

    backend = OpenAICompatBackend(
        "http://127.0.0.1:9/v1", retry=RetryPolicy(max_attempts=1), timeout_s=1
    )
    checks = run_checks(backend)
    assert len(checks) == 1 and not checks[0].ok and "base_url" in checks[0].advice


@pytest.mark.data
def test_code_task_on_dev_is_measured_on_test_inputs(tmp_path, capsys):
    """Task 8's dev inputs carry no wrapper signature, so nothing there can be executed.
    Asking for dev must measure on test inputs instead, whatever script calls the CLI."""
    run_dir = tmp_path / "run"
    code = main(["--config", "configs/offline.yaml", "run", "--task", "8", "--split", "dev",
                 "--limit", "2", "--run-dir", str(run_dir)])  # fmt: skip
    assert code == 0
    assert "measuring on the first 2 TEST inputs" in capsys.readouterr().out
    report = json.loads((run_dir / "task8" / "dev.report.json").read_text())
    assert report["measured_on"] == "test" and report["code_quality"]["n"] == 2


def test_dev_audit_rejects_wrong_scores_and_duplicate_ids(tmp_path, monkeypatch):
    import json

    import pytest

    from cairn.data import TaskData
    from cairn.eval import audit
    from cairn.types import Example, Prediction

    data = TaskData(1, "small", "", (Example("q", "input", "label"),), ())
    monkeypatch.setattr(audit, "load_task", lambda *_: data)
    row = json.dumps(Prediction("q", "label").to_json())
    for i in range(1, 8):
        folder = tmp_path / f"task{i}"
        folder.mkdir()
        report = {
            "task": i,
            "split": "dev",
            "n": 1,
            "config": {
                "split_seed": 42,
                "splits": {"calib": 0, "dev": 1, "holdout": 0, "max_reserved_frac": 1.0},
            },
            "eval": {"correct": 1, "n": 1, "accuracy": 1.0},
        }
        (folder / "dev.report.json").write_text(json.dumps(report), encoding="utf-8")
        (folder / "dev.journal.jsonl").write_text(row, encoding="utf-8")
    assert audit.audit_dev(tmp_path)["correct"] == 7
    journal = folder / "dev.journal.jsonl"
    journal.write_text(row + "\n" + row, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        audit.audit_dev(tmp_path)
    journal.write_text(json.dumps(Prediction("q", "bad").to_json()), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        audit.audit_dev(tmp_path)
