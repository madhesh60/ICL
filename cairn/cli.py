"""Command line entry point: `cairn <command>`.

cairn inspect                 what is in the data, how it is split, token sizes
cairn context --task 6        measure a prompt design without a model
cairn baseline                scores that need no model (reference points)
cairn doctor                  check what the model server supports
cairn run --task all          label dev or test inputs
cairn tune --task 5           race pipeline settings on the dev split
cairn submit                  validate and zip the eight prediction files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from cairn.config import AppConfig, BackendConfig, load_config
from cairn.context.builder import ContextBuilder
from cairn.data import REPO_ROOT, load_task, split_examples
from cairn.llm import CachedBackend, DemoReaderBackend, LLMBackend, OpenAICompatBackend, RetryPolicy
from cairn.pipeline import Journal, RunConfig, TaskRunner
from cairn.tasks import get_task
from cairn.tokens import get_counter

log = logging.getLogger("cairn")


def _task_ids(value: str) -> list[int]:
    if value == "all":
        return list(range(1, 9))
    ids = [int(part) for part in value.split(",")]
    if not all(1 <= i <= 8 for i in ids):
        raise argparse.ArgumentTypeError("task must be 1..8, a comma list, or 'all'")
    return ids


def make_backend(cfg: BackendConfig, cache_path: Path | None = None) -> LLMBackend:
    if cfg.kind == "demo-reader":
        return DemoReaderBackend()
    if cfg.kind != "openai":
        raise SystemExit(f"unknown backend kind: {cfg.kind!r} (use 'openai' or 'demo-reader')")
    backend: LLMBackend = OpenAICompatBackend(
        cfg.base_url,
        cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        retry=RetryPolicy(max_attempts=cfg.max_attempts),
        structured_api=cfg.structured_api,
        thinking_control=cfg.thinking_control,
    )
    if cfg.response_cache and cache_path is not None:
        backend = CachedBackend(backend, cache_path)
    return backend


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------------- commands


def cmd_inspect(args: argparse.Namespace, app: AppConfig) -> int:
    counter = get_counter(args.tokenizer)
    print(f"token counter: {counter.name} (safety margin {counter.margin:.0%})\n")
    print(
        "| task | name | kind | examples | pool | calib | dev | holdout | test | tokens/example | labels |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for task_id in _task_ids(args.task):
        task = load_task(task_id, args.data_dir)
        run_cfg = app.for_task(task_id)
        splits = split_examples(task.examples, run_cfg.splits, run_cfg.split_seed)
        spec = get_task(task_id, tuple(e.output for e in splits.pool))
        sample = task.examples[:: max(1, len(task.examples) // 300)]
        per = sum(counter.count(e.input) + counter.count(e.output) for e in sample) / len(sample)
        labels = (
            ", ".join(spec.labels)
            if spec.labels
            else f"{len({e.output for e in task.examples})} distinct"
        )
        print(
            f"| {task_id} | {spec.name} | {spec.kind} | {len(task.examples)} | {len(splits.pool)} | "
            f"{len(splits.calib)} | {len(splits.dev)} | {len(splits.holdout)} | {len(task.test)} | "
            f"{per:.0f} | {labels} |"
        )
    return 0


def cmd_context(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.eval.contextbench import context_report, simulate_prefix_cache

    counter = get_counter(args.tokenizer)
    for task_id in _task_ids(args.task):
        task = load_task(task_id, args.data_dir)
        run_cfg = app.for_task(task_id)
        splits = split_examples(task.examples, run_cfg.splits, run_cfg.split_seed)
        spec = get_task(task_id, tuple(e.output for e in splits.pool))
        queries = splits.as_queries("dev")[: args.limit]
        context_cfg = run_cfg.context
        if args.clusters is not None:
            context_cfg = replace(context_cfg, n_clusters=args.clusters)
        if args.selection is not None:
            context_cfg = replace(context_cfg, selection=args.selection)
        builder = ContextBuilder(
            spec, splits.pool, counter, context_cfg,
            definition=task.definition, target_inputs=[q.input for q in queries],
        )  # fmt: skip
        report = context_report(builder, queries)
        grouped = simulate_prefix_cache(builder, queries, counter, grouped=True, slots=args.slots)
        arrival = simulate_prefix_cache(builder, queries, counter, grouped=False, slots=args.slots)
        print(
            f"task {task_id} ({spec.name}), {len(queries)} dev inputs, {builder.n_clusters} cluster(s)"
        )
        print("  context :", json.dumps(report.to_json()))
        print("  cache, grouped order :", json.dumps(grouped.to_json()))
        print("  cache, arrival order :", json.dumps(arrival.to_json()))
        if args.show:
            for message in builder.build(queries[0]).messages:
                content = message["content"]
                body = (
                    content
                    if len(content) < 1600
                    else f"{content[:800]}\n  [...]\n{content[-700:]}"
                )
                print(f"\n--- {message['role']} ({counter.count(content)} tokens) ---\n{body}")
    return 0


def cmd_baseline(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.eval.baselines import knn_accuracy, majority_accuracy

    counter = get_counter(args.tokenizer)
    print("| task | name | dev inputs | majority label | majority accuracy | kNN-vote accuracy |")
    print("|---|---|---|---|---|---|")
    for task_id in _task_ids(args.task):
        if task_id == 8:
            continue  # code has no single right string to compare with
        task = load_task(task_id, args.data_dir)
        run_cfg = app.for_task(task_id)
        splits = split_examples(task.examples, run_cfg.splits, run_cfg.split_seed)
        spec = get_task(task_id, tuple(e.output for e in splits.pool))
        dev = splits.as_queries("dev")
        context = run_cfg.context
        if args.neighbors is not None:
            context = replace(context, structural_neighbors=args.neighbors == "structural")
            if context.structural_neighbors and spec.structure_features is None:
                raise ValueError(f"task {task_id} has no structural retrieval features")
        builder = ContextBuilder(spec, splits.pool, counter, context, definition=task.definition)
        label, majority = majority_accuracy(splits.pool, dev)
        shown = label if len(label) <= 12 else label[:12] + "..."
        print(
            f"| {task_id} | {spec.name} | {len(dev)} | {shown} | {majority:.1%} | "
            f"{knn_accuracy(builder, dev, args.k):.1%} |"
        )
    return 0


def cmd_doctor(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.doctor import run_checks

    backend = make_backend(replace(app.backend, response_cache=False))
    if not isinstance(backend, OpenAICompatBackend):
        print("doctor only applies to the 'openai' backend kind")
        return 1
    backend.retry = RetryPolicy(max_attempts=2, base_delay_s=0.5)
    print(f"checking {backend.base_url} (model {backend.model})\n")
    failed = 0
    for check in run_checks(backend):
        print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        if check.advice:
            print(f"       -> {check.advice}")
        failed += not check.ok
    return 1 if failed else 0


def _run_one(
    task_id: int, args: argparse.Namespace, app: AppConfig, run_dir: Path
) -> dict[str, object]:
    from cairn.eval.metrics import evaluate
    from cairn.submit import write_task_file

    task = load_task(task_id, args.data_dir)
    run_cfg: RunConfig = app.for_task(task_id)
    if args.workers is not None:
        run_cfg = replace(run_cfg, workers=args.workers)
    task_dir = run_dir / f"task{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    backend = make_backend(app.backend, task_dir / "responses.sqlite")
    runner = TaskRunner(task, backend, get_counter(args.tokenizer), run_cfg)
    queries = list(task.test) if args.split == "test" else runner.splits.as_queries(args.split)
    measured_on = args.split
    if runner.spec.kind == "code" and args.split != "test":
        from cairn.tasks.kernel import parse_wrapper_signature

        if not any(parse_wrapper_signature(q.input) for q in queries):
            # Task 8's demonstrations describe existing code and carry no wrapper signature
            # (0 of 184), so a dev input can never be executed (E001 and E002: 24/24
            # "not_run").  The code metric needs no gold answer, so measure on test inputs.
            queries = list(task.test)
            measured_on = "test"
            print(
                f"task {task_id}: {args.split} inputs have no wrapper signature to execute; "
                f"measuring on the first {args.limit or len(queries)} TEST inputs instead"
            )
    queries = queries[: args.limit]
    started = time.time()
    journal = _open_journal(task_dir, args.split, run_cfg, backend.model, fresh=args.fresh)
    predictions = runner.run(queries, journal)
    summary: dict[str, object] = {
        "task": task_id,
        "split": args.split,
        "measured_on": measured_on,
        "n": len(predictions),
        "seconds": round(time.time() - started, 1),
        "sources": _count(p.source for p in predictions),
        "notes": runner.notes,
        "usage": runner.meter.summary(),
        "config": asdict(run_cfg),
        "commit": _git_commit(),
        "model": backend.model,
    }
    if args.split == "test":
        write_task_file(task_dir / f"openseek-{task_id}.jsonl", predictions)
    if runner.spec.kind == "code":
        # Code has no single right string, so exact match says nothing.  Report how far
        # each kernel got instead, and the competition's consistency score.  This needs no
        # gold answer, so it is measured on the TEST inputs too, and for task 8 it must be:
        # the demonstration kernels carry no wrapper signature (0 of 184), so on the dev
        # split nothing can ever be executed (E001: 24 of 24 "not_run").
        from cairn.tasks.kernel_exec import code_quality

        quality = code_quality(predictions)
        summary["code_quality"] = quality
        print(
            f"task {task_id} [{args.split}] static check OK {quality['static_ok']:.0%}, "
            f"runs {quality['callable']:.0%}, matches PyTorch {quality['matches_torch']:.0%}, "
            f"consistency score {quality['consistency']:.3f} (n={quality['n']})"
        )
        print(f"    execution outcomes: {quality['statuses']}")
    elif args.split != "test":
        report = evaluate(predictions, queries)
        summary["eval"] = report.to_json()
        print(f"task {task_id} [{args.split}] {report.line()}")
        for source, row in report.by_source.items():
            print(f"    {source:16s} {row['share']:.0%} of inputs, accuracy {row['accuracy']:.1%}")
    _print_preparation_notes(runner.notes)
    (task_dir / f"{args.split}.report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def _print_preparation_notes(notes: dict[str, object]) -> None:
    """One line per preparation decision, so a log alone tells what the run relied on."""
    program = notes.get("program")
    if isinstance(program, dict):
        if program.get("accepted"):
            print(
                f"    program: ACCEPTED after {program['attempts']} attempt(s); verified on "
                f"{program['verified_on']} + {program['confirmed_on']} held-out examples"
            )
        else:
            print(
                f"    program: not accepted after {program['attempts']} attempt(s), best verify "
                f"accuracy {program['best_verify_accuracy']:.1%}; answers came from the model"
            )
            for row in program.get("attempt_log", [])[:6]:  # type: ignore[union-attr]
                print(
                    f"        round {row['round']} thinking={row['thinking']} finish={row['finish'] or '?'} "
                    f"verify={row['verify_accuracy']:.1%} {row['problem'][:90]}"
                )
    elif isinstance(program, str):
        print(f"    program: {program}")
    policy = notes.get("policy")
    if isinstance(policy, dict):
        print(
            f"    policy: {policy['chosen'].upper()} from a probe on {policy['probe_examples']} "
            f"held-out inputs: cheap {policy['cheap_accuracy']:.0%}, thinking "
            f"{policy['costly_accuracy']:.0%}"
        )
        for name in ("all", "sure", "unsure"):
            row = policy[name]
            if row["n"]:
                print(
                    f"        {name:6s} n={row['n']:3d} cheap {row['cheap_accuracy']:.0%} vs thinking "
                    f"{row['costly_accuracy']:.0%} (only-cheap {row['only_cheap_right']}, "
                    f"only-thinking {row['only_costly_right']}, p={row['p_value']:.3f})"
                )
    elif isinstance(policy, str):
        print(f"    policy: {policy}")
    calibration = notes.get("calibration")
    if isinstance(calibration, dict):
        line = (
            f"    calibration: {calibration['method']} on {calibration['fitted_on']} examples, "
            f"held-out accuracy {calibration['accuracy_before']:.1%} -> {calibration['accuracy_after']:.1%}"
        )
        conformal = calibration.get("conformal")
        if isinstance(conformal, dict):
            line += (
                f"; sure on {conformal['singleton_rate']:.0%} of held-out inputs, right on "
                f"{conformal['singleton_accuracy']:.0%} of those"
                + (
                    ""
                    if conformal.get("singletons_trusted", True)
                    else " -> NOT trusted, all inputs escalate"
                )
            )
        print(line)
    elif isinstance(calibration, str):
        print(f"    calibration: {calibration}")


def _open_journal(
    task_dir: Path, split: str, run_cfg: RunConfig, model: str, *, fresh: bool
) -> Journal:
    """Open the resume journal, refusing to mix answers made with different settings.

    Resuming is only correct if the settings are the same as when the journal was
    started.  A fingerprint of the settings is stored next to the journal; if it
    does not match, we stop instead of quietly mixing old and new answers.
    """
    path = task_dir / f"{split}.journal.jsonl"
    meta = task_dir / f"{split}.journal.meta.json"
    payload = json.dumps({"config": asdict(run_cfg), "model": model}, sort_keys=True, default=str)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
    if fresh:
        path.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
    if path.exists() and meta.exists():
        stored = json.loads(meta.read_text(encoding="utf-8")).get("fingerprint")
        if stored != fingerprint:
            raise SystemExit(
                f"{path} was started with different settings (fingerprint {stored} vs "
                f"{fingerprint}). Use a new --run-dir, or add --fresh to start this task over."
            )
    meta.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    return Journal(path)


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def cmd_run(args: argparse.Namespace, app: AppConfig) -> int:
    run_dir = Path(args.run_dir)
    for task_id in _task_ids(args.task):
        summary = _run_one(task_id, args, app, run_dir)
        if args.split == "test":
            print(
                f"task {task_id} [test] wrote {summary['n']} predictions, sources: {summary['sources']}"
            )
    print(f"\nrun folder: {run_dir}")
    return 0


def cmd_tune(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.eval.autotune import successive_halving

    counter = get_counter(args.tokenizer)
    run_dir = Path(args.run_dir)
    for task_id in _task_ids(args.task):
        task = load_task(task_id, args.data_dir)
        base = app.for_task(task_id)
        candidates = {
            "base": base,
            "no-escalation": replace(base, strategy=replace(base.strategy, escalate=False)),
            "tail-24": replace(base, context=replace(base.context, tail_k=24)),
            "tail-as-text": replace(base, context=replace(base.context, tail_as_turns=False)),
            "random-bulk": replace(base, context=replace(base.context, selection="random")),
            "clusters-4": replace(base, context=replace(base.context, n_clusters=4)),
        }
        cache = run_dir / f"task{task_id}" / "responses.sqlite"
        cache.parent.mkdir(parents=True, exist_ok=True)
        backend = make_backend(app.backend, cache)

        def factory(cfg: RunConfig, task=task, backend=backend):
            return lambda: TaskRunner(task, backend, counter, cfg)

        dev = TaskRunner(task, backend, counter, base).splits.as_queries("dev")[: args.limit]
        result = successive_halving(
            {n: factory(c) for n, c in candidates.items()}, dev, start=args.start
        )
        print(f"\ntask {task_id}: winner = {result.winner}")
        if result.runner_up is not None:
            print(
                f"paired test against runner-up '{result.runner_up}' on the "
                f"{result.compared_on} examples both labelled: p = {result.p_value:.3f} "
                "(below 0.05 = the difference is unlikely to be luck)"
            )
        print(result.table())
    return 0


def cmd_bench(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.eval.bench import run_bench

    report = run_bench(app, get_counter(args.tokenizer), args.data_dir, dev_limit=args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwritten to {out}")
    return 0


def cmd_submit(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.submit import SubmissionError, build_zip

    try:
        out = build_zip(args.run_dir, args.out, args.data_dir)
    except SubmissionError as err:
        print(err, file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def cmd_replay_kernels(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.eval.replay import replay_kernels

    try:
        report = replay_kernels(
            Path(args.predictions), load_task(8, args.data_dir), execute=args.execute
        )
    except (OSError, ValueError) as err:
        print(err, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


def cmd_audit(args: argparse.Namespace, app: AppConfig) -> int:
    from cairn.eval.audit import audit_dev

    try:
        report = audit_dev(Path(args.run_dir), args.data_dir)
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(err, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cairn", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", help="YAML settings file (default: built-in settings)")
    parser.add_argument("--data-dir", help="folder with the 8 task json files")
    parser.add_argument("--tokenizer", help="local Qwen3-4B folder for exact token counts")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(fn=fn)
        return p

    p = add("inspect", cmd_inspect, "show data, splits and token sizes")
    p.add_argument("--task", default="all")

    p = add("context", cmd_context, "measure a prompt design without a model")
    p.add_argument("--task", default="6")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--clusters", type=int)
    p.add_argument("--selection", choices=["coverage", "random", "first"])
    p.add_argument(
        "--slots", type=int, default=2, help="prompt beginnings the simulated server remembers"
    )
    p.add_argument("--show", action="store_true", help="print one assembled prompt (shortened)")

    p = add("baseline", cmd_baseline, "model-free reference scores on the dev split")
    p.add_argument("--task", default="all")
    p.add_argument("-k", type=int, default=15)
    p.add_argument(
        "--neighbors",
        choices=["lexical", "structural"],
        help="override the configured retrieval method",
    )

    add("doctor", cmd_doctor, "check which features the model server supports")

    p = add("bench", cmd_bench, "measure everything that needs no model; write a report")
    p.add_argument("--limit", type=int, default=300, help="dev inputs per task")
    p.add_argument("--out", default="docs/results/offline_benchmarks.md")

    p = add("run", cmd_run, "label inputs")
    p.add_argument("--task", default="all")
    p.add_argument("--split", default="dev", choices=["dev", "holdout", "calib", "test"])
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--fresh", action="store_true", help="discard saved answers and start over")

    p = add("tune", cmd_tune, "race pipeline settings on the dev split")
    p.add_argument("--task", default="5")
    p.add_argument("--limit", type=int, default=160)
    p.add_argument("--start", type=int, default=40)
    p.add_argument("--run-dir", default="runs/tune")

    p = add(
        "replay-kernels", cmd_replay_kernels, "check saved task-8 kernels without a model server"
    )
    p.add_argument("predictions", help="official-format openseek-8.jsonl (a subset is allowed)")
    p.add_argument("--execute", action="store_true", help="also run the kernels in child processes")

    p = add(
        "audit-dev",
        cmd_audit,
        "rescore saved journals for tasks 1-7 against the official development IDs",
    )
    p.add_argument("--run-dir", required=True)

    p = add("submit", cmd_submit, "validate and zip the eight prediction files")
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--out", default="submission.zip")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = load_config(args.config)
    return int(args.fn(args, app))


if __name__ == "__main__":
    raise SystemExit(main())
