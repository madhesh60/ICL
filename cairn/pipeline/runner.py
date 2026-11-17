"""TaskRunner: prepares one task and labels a list of inputs.

`prepare()` is where the left-over labelled examples are put to work, before a
single test input is touched:

    pool      -> chosen and arranged into the long prompt
    calib     -> half A fits the calibrator (removes the model's label lean)
                 half B fits the conformal threshold (decides when to escalate)
                 for exact-procedure tasks: tests and confirms the induced program
                 for guidelines: decides whether the induced rulebook is kept
    dev       -> never used here; it stays clean for honest evaluation

`run()` then labels inputs with a crash-safe journal and cache-aware scheduling.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cairn.context.builder import ContextBuilder, ContextConfig
from cairn.context.render import render_demo
from cairn.data import SplitSizes, TaskData, split_examples
from cairn.decoding.calibrate import (
    BatchCalibrator,
    IdentityCalibrator,
    StackedCalibrator,
    VectorScaling,
)
from cairn.decoding.conformal import SplitConformal
from cairn.decoding.score import distinct_starts, score_options
from cairn.eval.metrics import EvalReport, evaluate
from cairn.eval.stats import mcnemar_exact
from cairn.induction.guideline import GuidelineInducer
from cairn.induction.program import InducedProgram, ProgramInducer
from cairn.llm.base import LLMBackend, LLMError
from cairn.pipeline.annotator import Annotator, StrategyConfig
from cairn.pipeline.journal import Journal
from cairn.pipeline.memory import AnnotationMemory
from cairn.pipeline.meter import UsageMeter
from cairn.pipeline.scheduler import run_grouped
from cairn.pipeline.signals import signal_row
from cairn.tasks import get_task
from cairn.tokens import TokenCounter
from cairn.types import Example, Prediction, Query

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    context: ContextConfig = field(default_factory=ContextConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    splits: SplitSizes = field(default_factory=SplitSizes)
    split_seed: int = 42
    workers: int = 8
    # How many held-out examples the preparation steps may use.
    calib_limit: int = 300
    # Session memory for continuous annotation (off by default).
    memory: bool = False
    memory_wave: int = 64
    memory_max_items: int = 48
    memory_min_confidence: float = 0.95


def _chunks(items: Sequence[Query], size: int) -> list[Sequence[Query]]:
    return [items[i : i + size] for i in range(0, len(items), size)] or [items]


class TaskRunner:
    def __init__(
        self,
        task: TaskData,
        backend: LLMBackend,
        counter: TokenCounter,
        config: RunConfig | None = None,
    ) -> None:
        self.task = task
        self.config = config or RunConfig()
        self.counter = counter
        self.meter = UsageMeter(backend)
        self.splits = split_examples(task.examples, self.config.splits, self.config.split_seed)
        labels = tuple(e.output for e in self.splits.pool)
        self.spec = get_task(task.task_id, labels)
        self.builder: ContextBuilder | None = None
        self.calibrator: object = IdentityCalibrator()
        self.conformal: SplitConformal | None = None
        self.program: InducedProgram | None = None
        self.memory: AnnotationMemory | None = None
        self.signal_fn = None
        # Set to False by prepare() when held-out examples show the cheap scoring rung
        # cannot be trusted when it claims to be sure.
        self.trust_scores = True
        # Everything prepare() decided, in plain data, for the run report.
        self.notes: dict[str, object] = {}

    # ----------------------------------------------------------------- prepare

    def prepare(self, queries: Sequence[Query]) -> None:
        cfg = self.config
        self.builder = ContextBuilder(
            self.spec,
            self.splits.pool,
            self.counter,
            cfg.context,
            definition=self.task.definition,
            target_inputs=[q.input for q in queries],
        )
        self.notes["pool"] = len(self.splits.pool)
        self.notes["clusters"] = self.builder.n_clusters
        calib = list(self.splits.calib[: cfg.calib_limit])

        if cfg.strategy.induce_guidelines and self.spec.kind in {"classification", "generation"}:
            self._prepare_guidelines(calib)
        if self.spec.kind == "deterministic" and cfg.strategy.use_program:
            self._prepare_program(calib)
        if self.spec.kind == "classification" and cfg.strategy.use_scoring:
            self._prepare_calibration(calib)
        if cfg.memory:
            self.memory = AnnotationMemory(
                max_items=cfg.memory_max_items, min_confidence=cfg.memory_min_confidence
            )

    def _prepare_program(self, calib: list[Example]) -> None:
        assert self.builder is not None
        if len(calib) < 20:
            self.notes["program"] = "skipped: too few held-out examples to verify a program"
            return
        half = len(calib) // 2
        cfg = self.config.strategy
        inducer = ProgramInducer(
            self.meter,
            max_rounds=cfg.induction_rounds,
            max_tokens=cfg.induction_max_tokens,
            seed=cfg.seed,
        )
        # A short, varied slice of the shared block: evenly spaced, so inputs of every
        # length are represented.  The first real run sent the whole 30K-token block with a
        # 2K reply budget and never got a program back.
        bulk = self.builder.bulk_examples(0)
        step = max(1, len(bulk) // max(1, cfg.induction_examples))
        shown = bulk[::step][: cfg.induction_examples]
        examples_text = "# Examples\n\n" + "".join(render_demo(self.spec, e) for e in shown)
        result = inducer.induce(self.task.definition, examples_text, calib[:half], calib[half:])
        self.program = result.program
        self.notes["program"] = {
            "accepted": result.accepted,
            "attempts": len(result.attempts),
            "examples_shown": len(shown),
            "verified_on": half,
            "confirmed_on": len(calib) - half,
            "best_verify_accuracy": max((a.verify_accuracy for a in result.attempts), default=0.0),
            "attempt_log": [
                {
                    "round": a.round,
                    "thinking": a.thinking,
                    "finish": a.finish,
                    "verify_accuracy": round(a.verify_accuracy, 4),
                    "problem": (a.problems[0][:200] if a.problems else ""),
                }
                for a in result.attempts
            ],
            "code": result.program.code if result.program else None,
        }
        if result.accepted:
            log.info(
                "task %d: program accepted after %d attempt(s)",
                self.task.task_id,
                len(result.attempts),
            )
        else:
            log.warning(
                "task %d: no program accepted after %d attempt(s) (best verify accuracy %.1f%%); "
                "falling back to in-context labelling",
                self.task.task_id,
                len(result.attempts),
                100 * self.notes["program"]["best_verify_accuracy"],  # type: ignore[index]
            )

    def _score_examples(
        self, builder: ContextBuilder, examples: Sequence[Example]
    ) -> np.ndarray | None:
        """Raw label log-probabilities for labelled held-out examples (one cheap call each)."""
        labels = self.spec.labels
        queries = [Query(e.id, e.input, e.output) for e in examples]

        def work(query: Query) -> np.ndarray:
            return score_options(
                self.meter, builder.build(query).messages, labels, tag="calibrate"
            )[0]

        try:
            rows = run_grouped(
                queries, lambda q: builder.route(q.input), work, workers=self.config.workers
            )
        except (ValueError, LLMError) as err:
            log.warning("could not score calibration examples: %s", err)
            return None
        return np.vstack(rows)

    def _prepare_calibration(self, calib: list[Example]) -> None:
        assert self.builder is not None
        cfg = self.config.strategy
        labels = self.spec.labels
        if not labels or not distinct_starts(labels):
            self.notes["calibration"] = "skipped: labels cannot be told apart by their first token"
            return
        if len(calib) < 40 or cfg.calibration == "none":
            self.notes["calibration"] = "none"
            return
        raw = self._score_examples(self.builder, calib)
        if raw is None:
            self.notes["calibration"] = "skipped: the server returned no probabilities"
            return
        index = {label: i for i, label in enumerate(labels)}
        y = np.array([index[e.output] for e in calib])
        # Two halves on purpose.  The conformal promise only holds if its threshold is
        # measured on examples the calibrator has NOT been fitted to.
        half = len(calib) // 2
        extra = None
        if cfg.calibration == "stack":
            builder = self.builder
            extra = np.vstack([signal_row(builder, e.input) for e in calib])
            self.calibrator = StackedCalibrator().fit(raw[:half], y[:half], extra[:half])
            self.signal_fn = lambda text: signal_row(builder, text)
        elif cfg.calibration == "vector":
            self.calibrator = VectorScaling().fit(raw[:half], y[:half])
        else:
            self.calibrator = BatchCalibrator().fit(raw[:half])
        held_extra = extra[half:] if extra is not None else None
        probs = np.exp(self.calibrator.transform(raw[half:], held_extra))  # type: ignore[attr-defined]
        raw_accuracy = float((raw[half:].argmax(axis=1) == y[half:]).mean())
        note: dict[str, object] = {
            "method": cfg.calibration,
            "fitted_on": half,
            "accuracy_before": round(raw_accuracy, 4),
            "accuracy_after": round(float((probs.argmax(axis=1) == y[half:]).mean()), 4),
        }
        if cfg.conformal_alpha is not None:
            self.conformal = SplitConformal(cfg.conformal_alpha).fit(probs, y[half:])
            report = self.conformal.report(probs, y[half:])
            n_single = round(report.singleton_rate * (len(calib) - half))
            # The conformal promise is about coverage ("the truth is in the set 90% of the
            # time"), not about the accuracy of one-label sets.  Check that separately:
            # if the sure answers on held-out examples are not accurate, do not stop at them.
            judged = n_single >= cfg.min_singleton_count
            self.trust_scores = (
                not judged
            ) or report.singleton_accuracy >= cfg.min_singleton_accuracy
            note["conformal"] = {
                "alpha": cfg.conformal_alpha,
                "threshold": round(float(self.conformal.threshold or 0.0), 4),
                "measured_on": len(calib) - half,
                "singleton_rate": round(report.singleton_rate, 4),
                "singleton_count": n_single,
                "singleton_accuracy": round(report.singleton_accuracy, 4),
                "singletons_trusted": self.trust_scores,
            }
            if not self.trust_scores:
                log.warning(
                    "task %d: the cheap scoring rung was right on only %.0f%% of the %d held-out "
                    "inputs it was sure about (bar %.0f%%); every input will think and vote",
                    self.task.task_id,
                    100 * report.singleton_accuracy,
                    n_single,
                    100 * cfg.min_singleton_accuracy,
                )
        self.notes["calibration"] = note

    def _prepare_guidelines(self, calib: list[Example]) -> None:
        """Induce a rulebook, then keep it only if held-out examples say it does not hurt."""
        assert self.builder is not None
        rendered = [render_demo(self.spec, e) for e in self.builder.bulk_examples(0)]
        rulebook = GuidelineInducer(self.meter).induce(self.task.definition, rendered)
        if not rulebook:
            self.notes["guidelines"] = "none produced"
            return
        probe = [Query(e.id, e.input, e.output) for e in calib[:120]]
        cheap = StrategyConfig(escalate=False, use_scoring=False, seed=self.config.strategy.seed)
        with_rules = self.builder.with_guidelines(rulebook)
        try:
            plain = evaluate(
                self._annotate_all(Annotator(self.spec, self.builder, self.meter, cheap), probe),
                probe,
            )
            ruled = evaluate(
                self._annotate_all(Annotator(self.spec, with_rules, self.meter, cheap), probe),
                probe,
            )
        except LLMError as err:
            self.notes["guidelines"] = f"not tested ({err}); discarded"
            return
        only_plain, only_ruled, p_value = mcnemar_exact(plain.hits, ruled.hits)
        keep = ruled.accuracy >= plain.accuracy
        self.notes["guidelines"] = {
            "kept": keep,
            "accuracy_without": round(plain.accuracy, 4),
            "accuracy_with": round(ruled.accuracy, 4),
            "only_without_right": only_plain,
            "only_with_right": only_ruled,
            "p_value": round(p_value, 4),
            "text": rulebook,
        }
        if keep:
            self.builder = with_rules

    # --------------------------------------------------------------------- run

    def _annotate_all(self, annotator: Annotator, queries: Sequence[Query]) -> list[Prediction]:
        return run_grouped(
            queries,
            lambda q: annotator.builder.route(q.input),
            annotator.annotate,
            workers=self.config.workers,
        )

    def run(self, queries: Sequence[Query], journal: Journal | None = None) -> list[Prediction]:
        """Label `queries`.  With a journal, finished inputs are skipped on restart."""
        if self.builder is None:
            self.prepare(queries)
        assert self.builder is not None
        done = journal.load() if journal is not None else {}
        pending = [q for q in queries if q.id not in done]
        if done:
            log.info(
                "resuming: %d of %d inputs already done", len(queries) - len(pending), len(queries)
            )

        program_answers: dict[str, str | None] = {}
        if self.program is not None and pending:
            outputs = self.program.run([q.input for q in pending])
            program_answers = {q.id: out for q, out in zip(pending, outputs, strict=True)}

        annotator = Annotator(
            self.spec,
            self.builder,
            self.meter,
            self.config.strategy,
            calibrator=self.calibrator,
            conformal=self.conformal,
            program_answers=program_answers,
            signal_fn=self.signal_fn,
            memory=self.memory,
            trust_scores=self.trust_scores,
        )
        waves = _chunks(pending, self.config.memory_wave) if self.memory is not None else [pending]
        for wave in waves:
            if not wave:
                continue
            results = run_grouped(
                wave,
                lambda q: self.builder.route(q.input),  # type: ignore[union-attr]
                annotator.annotate,
                workers=self.config.workers,
                on_result=(lambda _q, p: journal.append(p)) if journal is not None else None,
            )
            for prediction in results:
                done[prediction.query_id] = prediction
            if self.memory is not None:
                self.memory.commit()
        return [done[q.id] for q in queries]

    def evaluate(
        self, role: str = "dev", limit: int | None = None
    ) -> tuple[EvalReport, list[Prediction]]:
        """Label one of our own held-out roles and score it."""
        queries = self.splits.as_queries(role)[:limit]
        if not queries:
            raise ValueError(f"split '{role}' is empty for task {self.task.task_id}")
        predictions = self.run(queries)
        return evaluate(predictions, queries), predictions
