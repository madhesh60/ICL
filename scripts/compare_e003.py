"""Compare exactly the saved E003 inputs, with a GPU replay of both kernel files."""

import argparse
import json
import os
from pathlib import Path

from cairn.config import load_config
from cairn.data import load_task, split_examples
from cairn.eval.metrics import is_correct
from cairn.eval.replay import replay_kernels
from cairn.eval.stats import mcnemar_exact
from cairn.tasks.kernel_exec import CALLABLE, PROVEN_GOOD

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "dev_results_now"


def rows_by_id(path, key, expected):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ident = row[key]
        if ident in rows:
            raise ValueError(f"{path}: duplicate ID {ident}")
        rows[ident] = row
    if rows.keys() != set(expected):
        missing = sorted(set(expected) - rows.keys())
        extra = sorted(rows.keys() - set(expected))
        raise ValueError(f"{path}: input mismatch. Missing {missing}, extra {extra}")
    return rows


def dev_queries(task_id):
    config = load_config(ROOT / "configs/default.yaml").for_task(task_id)
    task = load_task(task_id)
    return split_examples(task.examples, config.splits, config.split_seed).as_queries("dev")[:60]


def dev_hits(root, task_id, queries):
    folder = root / f"task{task_id}"
    rows = rows_by_id(folder / "dev.journal.jsonl", "query_id", [q.id for q in queries])
    hits = [is_correct(rows[q.id]["answer"], q.gold) for q in queries]
    report = json.loads((folder / "dev.report.json").read_text(encoding="utf-8"))
    if (report["model"], report["measured_on"], report["n"], report["eval"]["correct"]) != (
        "Qwen3-4B",
        "dev",
        60,
        sum(hits),
    ):
        raise ValueError(f"{folder}: recorded report disagrees with rescored journal")
    return hits


def paired(ids, before, after):
    regressed, gained, p_value = mcnemar_exact(before, after)
    return {
        "n": len(ids),
        "old_correct": sum(before),
        "new_correct": sum(after),
        "delta_pp": 100 * (gained - regressed) / len(ids),
        "gained_ids": [i for i, a, b in zip(ids, before, after, strict=True) if b and not a],
        "regressed_ids": [i for i, a, b in zip(ids, before, after, strict=True) if a and not b],
        "mcnemar_exact_p": p_value,
    }


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="fresh run to compare; omit for preflight")
    args = parser.parse_args()
    new = args.run_dir
    comparisons = {}
    for task_id, expected_count in [(2, 41), (7, 10)]:
        queries = dev_queries(task_id)
        ids = [q.id for q in queries]
        before = dev_hits(OLD, task_id, queries)
        if sum(before) != expected_count:
            raise ValueError(f"task {task_id}: unexpected E003 baseline {sum(before)}/60")
        if new is None:
            print(f"Task {task_id}: verified E003 {sum(before)}/{len(ids)} by ID")
            continue
        result = paired(ids, before, dev_hits(new, task_id, queries))
        comparisons[f"task{task_id}"] = result
        print(
            f"Task {task_id}: {sum(before)}/60 -> {result['new_correct']}/60 "
            f"({result['delta_pp']:+.1f} pp), {len(result['gained_ids'])} gained, "
            f"{len(result['regressed_ids'])} regressed, paired p={result['mcnemar_exact_p']:.4g}"
        )

    task = load_task(8)
    ids = [q.id for q in task.test[:24]]
    part = Path("task8/openseek-8.jsonl")
    rows_by_id(OLD / part, "test_sample_id", ids)
    if new is None:
        print("Task 8: verified all 24 E003 test IDs")
        return

    rows_by_id(new / part, "test_sample_id", ids)
    import torch

    if os.environ.get("CAIRN_TRITON_INTERPRET") != "0" or not torch.cuda.is_available():
        raise RuntimeError("The paired kernel replay requires CUDA and CAIRN_TRITON_INTERPRET=0")
    cases = []
    summaries = {}
    for label, root in [("old", OLD), ("new", new)]:
        print(f"Replaying {label} predictions on GPU...")
        replay = replay_kernels(root / part, task, execute=True)
        save(new / f"{label}-gpu-replay.json", replay)
        if replay["summary"]["statuses"].get("unavailable"):
            raise RuntimeError(f"{label} replay executor unavailable. See {label}-gpu-replay.json")
        by_id = replay["cases"]
        status = [by_id[i]["exec_status"] for i in ids]
        cases.append(status)
        summaries[label] = {
            "callable": sum(s in CALLABLE for s in status),
            "torch_matches": sum(s in PROVEN_GOOD for s in status),
            "static_ok": sum(by_id[i]["static_ok"] for i in ids),
            "consistency": replay["summary"]["consistency"],
        }
    before, after = summaries["old"], summaries["new"]
    delta = after["consistency"] - before["consistency"]
    comparisons["task8"] = {
        **summaries,
        "consistency_delta": delta,
        "numerical_matches": paired(ids, *[[s in PROVEN_GOOD for s in run] for run in cases]),
    }
    save(new / "comparison.json", comparisons)
    print(
        f"Task 8: callable {before['callable']}/24 -> {after['callable']}/24, "
        f"PyTorch matches {before['torch_matches']}/24 -> {after['torch_matches']}/24, "
        f"local consistency {before['consistency']:.4f} -> {after['consistency']:.4f} "
        f"({delta:+.4f})"
    )
    print(f"Saved {new / 'comparison.json'}")


if __name__ == "__main__":
    main()
