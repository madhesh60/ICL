"""Verify development run reports against their journals and the official split."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from cairn.data import SplitSizes, load_task, split_examples
from cairn.eval.metrics import evaluate, is_correct
from cairn.types import Prediction


def audit_dev(root: Path, data_dir: str | None = None) -> dict:
    tasks = {}
    for task_id in range(1, 8):
        folder = root / f"task{task_id}"
        report = json.loads((folder / "dev.report.json").read_text(encoding="utf-8"))
        if report["task"] != task_id or report.get("measured_on", report["split"]) != "dev":
            raise ValueError(f"task {task_id}: expected a development report")
        config = report["config"]
        split = split_examples(
            load_task(task_id, data_dir).examples,
            SplitSizes(**config["splits"]),
            config["split_seed"],
        )
        queries = split.as_queries("dev")[: report["n"]]
        # A journal may contain more rows after a longer run followed by a shorter one.
        # Only score the requested prefix, identified by the official IDs.
        saved = {}
        for line in (folder / "dev.journal.jsonl").read_text(encoding="utf-8").splitlines():
            p = Prediction.from_json(json.loads(line))
            if p.query_id in saved:
                raise ValueError(f"task {task_id}: duplicate journal ID {p.query_id}")
            saved[p.query_id] = p
        if not queries or len(queries) != report["n"] or any(q.id not in saved for q in queries):
            raise ValueError(
                f"task {task_id}: journal does not cover the reported development prefix"
            )
        predictions = [saved[q.id] for q in queries]
        result = evaluate(predictions, queries)
        if (
            result.correct != report["eval"]["correct"]
            or result.n != report["eval"]["n"]
            or round(result.accuracy, 4) != report["eval"]["accuracy"]
        ):
            raise ValueError(f"task {task_id}: recomputed score disagrees with report")
        lost_votes = sum(
            not hit and any(is_correct(a, q.gold) for a in p.meta.get("votes", {}))
            for q, p, hit in zip(queries, predictions, result.hits, strict=True)
        )
        tasks[task_id] = {
            "correct": result.correct,
            "n": result.n,
            "accuracy": result.accuracy,
            "by_source": result.by_source,
            "misses_with_gold_vote": lost_votes,
            "commit": report.get("commit"),
            "model": report.get("model"),
        }
    return {
        "tasks": tasks,
        "macro_accuracy": mean(t["accuracy"] for t in tasks.values()),
        "correct": sum(t["correct"] for t in tasks.values()),
        "n": sum(t["n"] for t in tasks.values()),
    }
