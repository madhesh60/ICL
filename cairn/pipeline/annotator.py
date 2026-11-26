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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cairn.context.builder import BuiltPrompt, ContextBuilder
from cairn.decoding.calibrate import IdentityCalibrator
from cairn.decoding.conformal import SplitConformal
from cairn.decoding.constraints import Regex, satisfies
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
    # "stack" also feeds cheap model-free signals (nearest-examples vote, task signals).
    calibration: Literal["none", "batch", "vector", "stack"] = "vector"
    # Chance we accept that the true label is outside the conformal set.  None = no
    # conformal gate; `accept_confidence` is used as a plain threshold instead.
    conformal_alpha: float | None = 0.1
    accept_confidence: float = 0.75
    # The cheap rung may only end the ladder if, on held-out examples, its "sure" answers
    # were right at least this often.  Measured on the second calibration half; needs at
    # least `min_singleton_count` sure answers there to be judged at all.  Below the bar,
    # every input goes to the expensive rung.  (First real run, task 2: the gate said
    # "sure" on 35% of inputs and was right on 29% of them.  A coverage promise is not an
    # accuracy promise; this check is.)
    min_singleton_accuracy: float = 0.85
    min_singleton_count: int = 10
    # Climb to the expensive rung when the cheap one is unsure.
    escalate: bool = True
    # Skip the cheap rung entirely: every input thinks and votes first.  Set by hand, or
    # chosen by the escalation probe below when held-out examples show the cheap rung is
    # confidently wrong (E001 task 2: four unanimous cheap samples, right 43% of the time).
    reason_first: bool = False
    # Escalation probe: before labelling, answer this many held-out calib examples BOTH
    # cheaply and expensively, then pick the policy from the measured, paired difference:
    #   cheap_only    the expensive rung does not beat the cheap one (ties go to cheap)
    #   ladder        it helps on the inputs the cheap rung is unsure about
    #   reason_first  it helps even where the cheap rung is sure
    # 0 = no probe; the settings above are used as written.
    escalation_probe: int = 0
    # Decomposition rung (tasks whose card defines it): the model lists the evidence in a
    # constrained format, code computes the answer, several samples vote.
    use_decompose: bool = True
    use_word_mask: bool = False
    use_clue_expansion: bool = False
    decompose_samples: int = 5
    decompose_temperature: float = 0.6
    # Run the whole cheap rung (direct answer AND the quick samples) but never think.  Used
    # by the probe, so that "sure" means what it means in the ladder, and by the cheap_only
    # policy, so that production answers exactly the way the probe measured.
    cheap_rung_only: bool = False
    # A policy that spends more compute must win a paired (McNemar) test at this level.
    probe_p_value: float = 0.10
    vote_n: int = 5
    think_max_tokens: int = 1024
    think_temperature: float = 0.6
    # Free-text tasks: extra cheap samples and the agreement needed to stop early.
    cheap_samples: int = 3
    agree_share: float = 0.67
    induce_guidelines: bool = False
    # Program induction (exact-procedure tasks).  The model sees a SHORT block of
    # examples, not the 30K-token annotation block: writing a five-line rule needs a
    # few dozen cases and a clear head, and the program is then proven on hundreds of
    # held-out examples anyway.  The budget covers thinking plus code; if a round still
    # yields no code, the next round runs without thinking.
    induction_examples: int = 48
    induction_max_tokens: int = 4096
    induction_rounds: int = 4
    # Code task.
    codegen_rounds: int = 3
    codegen_candidates: int = 2
    codegen_execute: bool = True
    # The task-8 score is S_exec^2 / S_call: a kernel that runs but computes the wrong
    # thing LOWERS it.  When execution proves the best candidate wrong (it disagrees with
    # PyTorch's own function, or returns NaN/inf), hand in a stub that fails loudly instead.
    abstain_on_proven_wrong: bool = True
    seed: int = 0


@dataclass
class _Candidate:
    code: str
    tier: int
    static_score: float
    feedback: list[str]
    exec_status: str
    static_ok: bool = True


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
        signal_fn: Callable[[str], np.ndarray] | None = None,
        trust_scores: bool = True,
    ) -> None:
        self.spec = spec
        self.builder = builder
        self.backend = backend
        self.config = config or StrategyConfig()
        self.calibrator = calibrator or IdentityCalibrator()
        self.conformal = conformal
        self.program_answers = program_answers or {}
        self.memory = memory
        self.signal_fn = signal_fn
        # False when held-out examples showed that the cheap rung's "sure" answers are
        # not reliable: then every input climbs to the expensive rung.
        self.trust_scores = trust_scores

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
        return majority_vote(
            self.backend.generate(request).texts,
            self._clean,
            allowed=allowed,
            group_key=self.spec.vote_key,
        )

    def is_sure(self, prediction: Prediction) -> bool:
        """Would the cheap rung have stopped the ladder on this answer?

        Used by the escalation probe to split held-out inputs into "the cheap rung was
        sure" and "it was not", because the value of thinking differs between the two.
        """
        if prediction.source == "score":
            plausible = prediction.meta.get("plausible", [])
            return len(plausible) == 1 and self.trust_scores
        if prediction.source in {"generate", "decompose"}:
            return prediction.confidence >= self.config.agree_share
        return prediction.source in {"program", "memory"}

    # ------------------------------------------------------------------ ladder

    def _climb(self, query: Query) -> Prediction:
        if self.spec.kind == "deterministic":
            answer = self.program_answers.get(query.id)
            if self._valid(answer):
                return Prediction(query.id, answer, 1.0, "program")  # type: ignore[arg-type]
        session = self.memory.session() if self.memory is not None else ()
        if self.spec.kind == "code":
            return self._codegen(query, session)
        if (
            self.config.use_decompose
            and not self.config.reason_first
            and self.spec.decompose is not None
            and self.spec.recompose is not None
        ):
            listed = self._decompose(query, session)
            if listed is not None:
                if not self.config.escalate or self.is_sure(listed):
                    return listed
                # The probe can select a ladder. An uncertain decomposition must
                # actually reach its measured expensive rung in production.
                prompt = self.builder.build(query, session)
                thought = self._think_vote(prompt)
                if thought.answer is None:
                    return listed
                meta = {
                    **self._meta(prompt),
                    "votes": thought.counts,
                    "cheap_votes": listed.meta["votes"],
                    "valid_votes": thought.valid,
                }
                return Prediction(
                    query.id,
                    thought.answer,
                    thought.share * thought.valid / self.config.vote_n,
                    "think_vote",
                    meta,
                )
        prompt = self.builder.build(query, session)
        if self.spec.kind == "classification":
            return self._classify(query, prompt)
        return self._free_form(query, prompt)

    def _decompose(self, query: Query, session: tuple) -> Prediction | None:
        """List the evidence, let code compute the answer, vote over samples.

        Returns None when the rung does not apply or nothing readable came back, so the
        normal ladder takes over.  If the server rejects the format constraint, the same
        request is sent once without it: a broken constraint must cost a retry, not the
        input (a rejected request would otherwise fall straight to the model-free answer).
        """
        assert self.spec.decompose is not None and self.spec.recompose is not None
        cfg = self.config
        mask = self.spec.word_mask if cfg.use_word_mask else None
        plan = (mask or self.spec.decompose)(query.input)
        if plan is None:
            return None
        instruction, pattern = plan
        rule = (
            "Return one binary digit per numbered word."
            if mask
            else "Return the requested list of words from the sentence."
        )
        prompt = self.builder.build(query, session, suffix=instruction, output_rule=rule)
        grammar = Regex(pattern)

        def ask(constraint: Regex | None) -> list[str]:
            request = GenRequest.from_messages(
                prompt.messages,
                max_tokens=96,
                temperature=cfg.decompose_temperature,
                top_p=0.95,
                top_k=20,
                n=cfg.decompose_samples,
                seed=cfg.seed,
                thinking=False,
                constraint=constraint,
                tag="decompose",
            )
            return self.backend.generate(request).texts

        try:
            replies = ask(grammar)
        except BadRequestError as err:
            log.warning("decompose constraint rejected (%s); retrying without it", err)
            replies = ask(None)
        recompose = self.spec.recompose

        def decode(reply: str) -> str | None:
            if mask:
                bits = reply.strip()
                return self._clean_value(str(bits.count("1"))) if satisfies(grammar, bits) else None
            return self._clean_value(recompose(reply))

        vote = majority_vote(replies, decode)
        if vote.answer is None:
            return None
        meta = self._meta(prompt)
        meta.update(
            votes=vote.counts, valid_votes=vote.valid, requested_votes=cfg.decompose_samples
        )
        meta["masks" if mask else "lists"] = [r[:160] for r in replies[:3]]
        support = vote.counts[vote.answer] / cfg.decompose_samples
        return Prediction(query.id, vote.answer, support, "decompose", meta)

    def _clean_value(self, answer: str | None) -> str | None:
        return answer if answer is not None and self._valid(answer) else None

    def _classify(self, query: Query, prompt: BuiltPrompt) -> Prediction:
        cfg, labels = self.config, self.spec.labels
        meta = self._meta(prompt)
        if cfg.reason_first or not (cfg.use_scoring and labels and distinct_starts(labels)):
            return self._free_form(query, prompt)
        try:
            raw, _ = score_options(self.backend, prompt.messages, labels)
        except ValueError:
            # The server gave no probabilities.  Generation still works.
            return self._free_form(query, prompt)
        extra = self.signal_fn(query.input) if self.signal_fn is not None else None
        probs = np.exp(self.calibrator.transform(raw, extra)[0])  # type: ignore[attr-defined]
        top = int(np.argmax(probs))
        if self.conformal is not None:
            keep = self.conformal.predict_sets(probs)[0]
        else:
            keep = [top] if probs[top] >= cfg.accept_confidence else []
        meta.update(probs=[round(float(p), 4) for p in probs], plausible=[labels[i] for i in keep])
        if not cfg.escalate or (len(keep) == 1 and self.trust_scores):
            return Prediction(query.id, labels[top], float(probs[top]), "score", meta)

        # Unsure.  Think and vote, restricted to the labels that are still plausible.
        # When the cheap rung is not trusted at all, its label sets are not either.
        allowed = tuple(labels[i] for i in keep) if len(keep) >= 2 and self.trust_scores else labels
        vote = self._think_vote(prompt, allowed)
        meta["votes"] = vote.counts
        if vote.answer is None:
            return Prediction(query.id, labels[top], float(probs[top]), "score", meta)
        return Prediction(query.id, vote.answer, vote.share, "think_vote", meta)

    def _clue_evidence(self, query: Query, prompt: BuiltPrompt) -> BuiltPrompt:
        """Ask Qwen for search terms. Only official examples enter the new reasoning tail."""
        session = self.memory.session() if self.memory else ()
        cue_prompt = self.builder.build(
            query,
            session,
            output_rule="Write up to twelve search words describing the expected answer type "
            "and the relationship in this clue. Omit guessed answers.",
        )
        request = GenRequest.from_messages(
            cue_prompt.messages,
            max_tokens=32,
            thinking=False,
            seed=self.config.seed,
            tag="clue_expansion",
        )
        try:
            texts = self.backend.generate(request).texts
        except BadRequestError:
            return prompt
        words = (texts[0] if texts else "").split()[:12]
        if not words:
            return prompt
        # Route and render the original clue. The expansion affects only the nearest examples.
        return self.builder.build(query, session, tail_query=query.input + " " + " ".join(words))

    def _free_form(self, query: Query, prompt: BuiltPrompt) -> Prediction:
        cfg = self.config
        meta = self._meta(prompt)
        replies: list[str] = []
        # Exact procedures always reason first (a glance cannot add 45 differences).  Other
        # tasks do so when configured, or when the escalation probe measured that they should.
        reason_first = self.spec.kind == "deterministic" or cfg.reason_first
        if not reason_first:
            replies.extend(a for a in self._direct(prompt) if a is not None)
            if (cfg.escalate or cfg.cheap_rung_only) and cfg.cheap_samples > 0:
                cheap = self._direct(
                    prompt, n=cfg.cheap_samples, temperature=0.7, tag="generate_sample"
                )
                replies.extend(a for a in cheap if a is not None)
            vote = majority_vote(replies, lambda a: a, group_key=self.spec.vote_key)
            meta["votes"] = vote.counts
            if vote.answer is not None and (not cfg.escalate or vote.share >= cfg.agree_share):
                return Prediction(query.id, vote.answer, vote.share, "generate", meta)
        if cfg.escalate or reason_first:
            if cfg.use_clue_expansion and self.spec.kind == "generation":
                original_tail = set(prompt.tail_ids)
                prompt = self._clue_evidence(query, prompt)
                meta.update(
                    self._meta(prompt), new_tail_examples=len(set(prompt.tail_ids) - original_tail)
                )
            thought = self._think_vote(prompt)
            meta.update(valid_think_votes=thought.valid, requested_think_votes=cfg.vote_n)
            # One count per reply, with the direct answers first so that they win ties.
            for answer, count in thought.counts.items():
                replies.extend([answer] * int(count))
            vote = majority_vote(replies, lambda a: a, group_key=self.spec.vote_key)
            meta["votes"] = vote.counts
            if vote.answer is not None:
                # Missing answers carry no support. One valid reply among five
                # should not populate session memory with confidence 1.0.
                support = vote.share * vote.valid / (vote.valid + cfg.vote_n - thought.valid)
                return Prediction(query.id, vote.answer, support, "think_vote", meta)
        if reason_first:
            if cfg.use_decompose and cfg.use_word_mask and self.spec.word_mask:
                recovered = self._decompose(query, self.memory.session() if self.memory else ())
                if recovered is not None:
                    recovered.meta.update(
                        valid_think_votes=0,
                        requested_think_votes=cfg.vote_n,
                        recovered_after="think_vote",
                    )
                    return recovered
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
            return _Candidate(code, 1, static.score, feedback, "not_run", static_ok=False)
        report: ExecReport | None = None
        if self.config.codegen_execute and signature is not None:
            report = execute_kernel(code, signature)
        if report is None or report.status in {
            "unavailable",
            "no_valid_inputs",
            "timeout",
            "insufficient_trials",
        }:
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
        seen_code: set[str] = set()
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
                if not code or code in seen_code:
                    continue
                seen_code.add(code)
                candidate = self._judge(code, signature)
                if best is None or (candidate.tier, candidate.static_ok, candidate.static_score) > (
                    best.tier,
                    best.static_ok,
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
            static_ok=best.static_ok,
            exec_status=best.exec_status,
            open_issues=best.feedback,
        )
        if cfg.abstain_on_proven_wrong and best.exec_status == "wrong_result":
            # Proven wrong after every repair round.  Handing it in would raise S_call
            # without raising S_exec, which lowers the score; a loud stub does neither.
            stub = self._fallback(query, "abstained: the best kernel was proven wrong")
            stub.meta.update(
                meta, static_ok=False, exec_status="abstained", abstained_code=best.code
            )
            stub.source = "abstain"
            return stub
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
