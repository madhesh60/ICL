"""The Annotator: decides, for one input, how much effort the answer needs.

Not every input deserves the same effort.  Most are easy and can be answered
with one cheap call.  A few are hard and deserve the expensive treatment
(thinking, several samples, a vote).  Spending the expensive treatment on
everything wastes compute; spending it on nothing loses accuracy.

So every task runs as a short ladder.  The answer leaves the ladder at the
first rung that is sure enough:

    classification   1. read the label from the model's probabilities   (1 token)
                     2. is the model *provably* sure?  -> done
                     3. otherwise think + vote, only among labels still plausible

    exact procedures 1. a program the model wrote and we verified       (no call)
                     2. otherwise think + vote
                     3. otherwise one constrained answer

    free text        1. one direct answer + a few cheap samples
                     2. do they agree?  -> done
                     3. otherwise think + vote over everything collected

    code             1. generate, with the required signature pinned
                     2. read it (static check) and run it (CPU interpreter)
                     3. show the model its own errors, let it repair, repeat

Whatever happens, an input never ends without an answer.  The last rung of
every ladder needs no model at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cairn.context.builder import BuiltPrompt, ContextBuilder
from cairn.decoding.calibrate import IdentityCalibrator
from cairn.decoding.conformal import SplitConformal
from cairn.decoding.constraints import satisfies
from cairn.decoding.parse import extract_code
from cairn.decoding.score import distinct_starts, score_options
from cairn.decoding.vote import VoteResult, majority_vote
from cairn.llm.base import BadRequestError, GenRequest, LLMBackend
from cairn.pipeline.memory import AnnotationMemory
from cairn.tasks.base import TaskSpec
from cairn.tasks.kernel import WrapperSignature, parse_wrapper_signature, static_check
from cairn.tasks.kernel_exec import ExecReport, execute_kernel
from cairn.types import Prediction, Query

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    # Exact-procedure tasks: let the model write a program and verify it.
    use_program: bool = True
    # Classification: read probabilities instead of generated text.
    use_scoring: bool = True
    calibration: Literal["none", "batch", "vector"] = "vector"
    # Chance we accept that the true label is outside the conformal set.  None = no
    # conformal gate; `accept_confidence` is used as a plain threshold instead.
    conformal_alpha: float | None = 0.1
    accept_confidence: float = 0.75
    # Climb to the expensive rung when the cheap one is unsure.
    escalate: bool = True
    vote_n: int = 5
    think_max_tokens: int = 1024
    think_temperature: float = 0.6
    # Free-text tasks: extra cheap samples and the agreement needed to stop early.
    cheap_samples: int = 3
    agree_share: float = 0.67
    induce_guidelines: bool = False
    # Code task.
    codegen_rounds: int = 3
    codegen_candidates: int = 2
    codegen_execute: bool = True
    seed: int = 0


@dataclass
class _Candidate:
    code: str
    tier: int
    static_score: float
    feedback: list[str]
    exec_status: str


# How far a code candidate got.  Higher is better.
_TIER_CONFIDENCE = {4: 0.95, 3: 0.7, 2: 0.5, 1: 0.2, 0: 0.0}


class Annotator:
    def __init__(
        self,
        spec: TaskSpec,
        builder: ContextBuilder,
        backend: LLMBackend,
        config: StrategyConfig | None = None,
        *,
        calibrator: object | None = None,
        conformal: SplitConformal | None = None,
        program_answers: dict[str, str | None] | None = None,
        memory: AnnotationMemory | None = None,
    ) -> None:
        self.spec = spec
        self.builder = builder
        self.backend = backend
        self.config = config or StrategyConfig()
        self.calibrator = calibrator or IdentityCalibrator()
        self.conformal = conformal
        self.program_answers = program_answers or {}
        self.memory = memory

    # ------------------------------------------------------------------ public

    def annotate(self, query: Query) -> Prediction:
        """Answer one input.  Always returns a usable answer.

        Error policy, in one place:
          * the request itself was rejected (BadRequestError)  -> this input gets
            the model-free fallback, the run goes on
          * the server is down (ServerUnavailableError)        -> NOT caught here;
            the run stops, the journal keeps what is done, and a restart resumes.
            Papering over a dead server with fallbacks would quietly ruin the run.
        """
        if self.memory is not None:
            remembered = self.memory.recall(query)
            if remembered is not None:
                return remembered
        try:
            prediction = self._climb(query)
        except BadRequestError as err:
            log.warning("request rejected for %s (%s); using the fallback", query.id, err)
            prediction = self._fallback(query, f"bad request: {err}")
        if not self._valid(prediction.answer):
            prediction = self._fallback(query, f"invalid answer from {prediction.source}")
        if self.memory is not None:
            self.memory.record(query, prediction)
        return prediction

    # ----------------------------------------------------------------- helpers

    def _valid(self, answer: str | None) -> bool:
        return answer is not None and satisfies(self.spec.constraint, answer)

    def _clean(self, reply: str) -> str | None:
        answer = self.spec.normalize(reply)
        return answer if self._valid(answer) else None

    def _meta(self, prompt: BuiltPrompt) -> dict[str, object]:
        return {
            "cluster": prompt.cluster,
            "demo_tokens": prompt.stats.unique_demo_tokens,
            "prompt_tokens": prompt.stats.prompt_tokens,
        }

    def _direct(
        self, prompt: BuiltPrompt, *, n: int = 1, temperature: float = 0.0, tag: str = "generate"
    ) -> list[str | None]:
        """Short, constrained answers without thinking."""
        request = GenRequest.from_messages(
            prompt.messages,
            max_tokens=self.spec.max_answer_tokens,
            temperature=temperature,
            top_p=0.8 if temperature > 0 else 1.0,
            top_k=20 if temperature > 0 else None,
            n=n,
            seed=self.config.seed,
            thinking=False,
            constraint=self.spec.constraint,
            tag=tag,
        )
        return [self._clean(text) for text in self.backend.generate(request).texts]

    def _think_vote(
        self, prompt: BuiltPrompt, allowed: tuple[str, ...] | None = None
    ) -> VoteResult:
        """Several reasoned answers, then a vote."""
        cfg = self.config
        request = GenRequest.from_messages(
            prompt.messages,
            max_tokens=cfg.think_max_tokens + self.spec.max_answer_tokens,
            temperature=cfg.think_temperature,
            top_p=0.95,
            top_k=20,
            n=cfg.vote_n,
            seed=cfg.seed,
            thinking=True,
            tag="think_vote",
        )
        return majority_vote(self.backend.generate(request).texts, self._clean, allowed=allowed)

    # ------------------------------------------------------------------ ladder

    def _climb(self, query: Query) -> Prediction:
        if self.spec.kind == "deterministic":
            answer = self.program_answers.get(query.id)
            if self._valid(answer):
                return Prediction(query.id, answer, 1.0, "program")  # type: ignore[arg-type]
        session = self.memory.session() if self.memory is not None else ()
        if self.spec.kind == "code":
            return self._codegen(query, session)
        prompt = self.builder.build(query, session)
        if self.spec.kind == "classification":
            return self._classify(query, prompt)
        return self._free_form(query, prompt)

    def _classify(self, query: Query, prompt: BuiltPrompt) -> Prediction:
        cfg, labels = self.config, self.spec.labels
        meta = self._meta(prompt)
        if not (cfg.use_scoring and labels and distinct_starts(labels)):
            return self._free_form(query, prompt)
        try:
            raw, _ = score_options(self.backend, prompt.messages, labels)
        except ValueError:
            # The server gave no probabilities.  Generation still works.
            return self._free_form(query, prompt)
        probs = np.exp(self.calibrator.transform(raw)[0])  # type: ignore[attr-defined]
        top = int(np.argmax(probs))
        if self.conformal is not None:
            keep = self.conformal.predict_sets(probs)[0]
        else:
            keep = [top] if probs[top] >= cfg.accept_confidence else []
        meta.update(probs=[round(float(p), 4) for p in probs], plausible=[labels[i] for i in keep])
        if len(keep) == 1 or not cfg.escalate:
            return Prediction(query.id, labels[top], float(probs[top]), "score", meta)

        # Unsure.  Think and vote, restricted to the labels that are still plausible.
        allowed = tuple(labels[i] for i in keep) if len(keep) >= 2 else labels
        vote = self._think_vote(prompt, allowed)
        meta["votes"] = vote.counts
        if vote.answer is None:
            return Prediction(query.id, labels[top], float(probs[top]), "score", meta)
        return Prediction(query.id, vote.answer, vote.share, "think_vote", meta)

    def _free_form(self, query: Query, prompt: BuiltPrompt) -> Prediction:
        cfg = self.config
        meta = self._meta(prompt)
        replies: list[str] = []
        reason_first = self.spec.kind == "deterministic"
        if not reason_first:
            replies.extend(a for a in self._direct(prompt) if a is not None)
            if cfg.escalate and cfg.cheap_samples > 0:
                cheap = self._direct(
                    prompt, n=cfg.cheap_samples, temperature=0.7, tag="generate_sample"
                )
                replies.extend(a for a in cheap if a is not None)
            vote = majority_vote(replies, lambda a: a)
            meta["votes"] = vote.counts
            if vote.answer is not None and (not cfg.escalate or vote.share >= cfg.agree_share):
                return Prediction(query.id, vote.answer, vote.share, "generate", meta)
        if cfg.escalate:
            thought = self._think_vote(prompt)
            # One count per reply, with the direct answers first so that they win ties.
            for answer, count in thought.counts.items():
                replies.extend([answer] * int(count))
            vote = majority_vote(replies, lambda a: a)
            meta["votes"] = vote.counts
            if vote.answer is not None:
                return Prediction(query.id, vote.answer, vote.share, "think_vote", meta)
        if reason_first:
            direct = [a for a in self._direct(prompt) if a is not None]
            if direct:
                return Prediction(query.id, direct[0], 0.5, "generate", meta)
        if replies:
            return Prediction(query.id, replies[0], 0.3, "generate", meta)
        return self._fallback(query, "no valid reply")

    # -------------------------------------------------------------------- code

    def _judge(self, code: str, signature: WrapperSignature | None) -> _Candidate:
        static = static_check(code, signature)
        feedback = list(static.problems)
        if not static.ok:
            return _Candidate(code, 1, static.score, feedback, "not_run")
        report: ExecReport | None = None
        if self.config.codegen_execute and signature is not None:
            report = execute_kernel(code, signature)
        if report is None or report.status in {"unavailable", "no_valid_inputs", "timeout"}:
            return _Candidate(
                code, 2, static.score, feedback, report.status if report else "not_run"
            )
        if report.status == "exec_ok":
            return _Candidate(code, 4, static.score, feedback, report.status)
        if report.status == "called_no_reference":
            return _Candidate(code, 3, static.score, feedback, report.status)
        hint = report.feedback()
        if report.status == "unconfirmed_error":
            # Every guessed input raised.  Maybe our guesses, maybe the code: worth one
            # repair attempt, but the candidate is not marked as broken.
            return _Candidate(code, 2, static.score, [hint], report.status)
        return _Candidate(code, 1, static.score, [*feedback, hint], report.status)

    def _codegen(self, query: Query, session: tuple) -> Prediction:
        cfg = self.config
        signature = parse_wrapper_signature(query.input)
        pin = ""
        if signature is not None:
            pin = (
                "\n\nThe wrapper function must be defined exactly like this:\n"
                f"{signature.render()}\n"
                "Launch the kernel with `kernel[grid](...)`. Use only functions that exist in "
                "triton.language. Do not use torch inside the @triton.jit kernel."
            )
        prompt = self.builder.build(query, session, suffix=pin)
        meta = self._meta(prompt)
        best: _Candidate | None = None
        extra: tuple[tuple[str, str], ...] = ()
        rounds_used = 0
        for round_index in range(max(1, cfg.codegen_rounds)):
            rounds_used = round_index + 1
            greedy = round_index == 0 and cfg.codegen_candidates == 1
            request = GenRequest(
                messages=tuple((m["role"], m["content"]) for m in prompt.messages) + extra,
                max_tokens=self.spec.max_answer_tokens,
                temperature=0.0 if greedy else 0.7,
                top_p=1.0 if greedy else 0.8,
                top_k=None if greedy else 20,
                n=cfg.codegen_candidates,
                seed=cfg.seed + round_index,
                thinking=False,
                tag="codegen" if round_index == 0 else "codegen_repair",
            )
            for reply in self.backend.generate(request).texts:
                code = extract_code(reply)
                if not code:
                    continue
                candidate = self._judge(code, signature)
                if best is None or (candidate.tier, candidate.static_score) > (
                    best.tier,
                    best.static_score,
                ):
                    best = candidate
            if best is not None and (best.tier >= 3 or not best.feedback):
                break  # proven to run, or nothing left that we could ask the model to fix
            if best is None:
                extra = (
                    ("user", "No Python code block was found. Reply with one complete code block."),
                )
            else:
                listing = "\n".join(f"- {line}" for line in best.feedback)
                extra = (
                    ("assistant", f"```python\n{best.code}\n```"),
                    (
                        "user",
                        f"That code has problems:\n{listing}\n\nFix them and reply with the complete "
                        "corrected code in one Python code block.",
                    ),
                )
        if best is None:
            return self._fallback(query, "no code produced")
        meta.update(
            rounds=rounds_used,
            tier=best.tier,
            exec_status=best.exec_status,
            open_issues=best.feedback,
        )
        return Prediction(query.id, best.code, _TIER_CONFIDENCE[best.tier], "codegen", meta)

    # ---------------------------------------------------------------- fallback

    def _fallback(self, query: Query, reason: str) -> Prediction:
        """The last rung: an answer made without the model.  Marked with confidence 0
        so that reports show exactly how often it was needed."""
        meta: dict[str, object] = {"reason": reason[:300]}
        if self.spec.kind == "code":
            signature = parse_wrapper_signature(query.input)
            header = signature.render() if signature else "def wrapper(*args, **kwargs):"
            stub = (
                "import torch\nimport triton\nimport triton.language as tl\n\n\n"
                f"{header}\n"
                '    raise NotImplementedError("no verified kernel could be generated")\n'
            )
            # A stub that fails loudly is honest: it cannot be mistaken for working code.
            return Prediction(query.id, stub, 0.0, "fallback_stub", meta)
        neighbours = self.builder.nearest(query.input, 9)
        outputs = [self.builder.pool[i].output for i, _ in neighbours]
        if self.spec.kind == "classification" and outputs:
            vote = majority_vote(outputs, lambda a: a)
            return Prediction(query.id, vote.answer or outputs[0], 0.0, "fallback_knn", meta)
        answer = outputs[0] if outputs else ""
        return Prediction(query.id, answer, 0.0, "fallback_nearest", meta)
