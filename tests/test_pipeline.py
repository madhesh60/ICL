import json
import threading
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from cairn.context import ContextBuilder, ContextConfig
from cairn.data import TaskData
from cairn.decoding.conformal import SplitConformal
from cairn.llm import (
    BadRequestError,
    DemoReaderBackend,
    FunctionBackend,
    GenResponse,
    ServerUnavailableError,
)
from cairn.pipeline import (
    AnnotationMemory,
    Annotator,
    Journal,
    RunConfig,
    StrategyConfig,
    TaskRunner,
    run_grouped,
)
from cairn.tasks import get_task
from cairn.types import Example, Prediction, Query
from tests.conftest import make_sentiment_pool

CTX = ContextConfig(min_demo_tokens=600, tail_k=4, tail_per_label=1)


def _spec_builder(counter):
    pool = make_sentiment_pool()
    spec = get_task(5, ("Not sad", "Sad"))
    return spec, ContextBuilder(spec, pool, counter, CTX, definition="sad?")


def _scorer(p_sad: float, think_answer: str = "Sad"):
    """A fake model: fixed label probabilities for scoring calls, fixed thinking answers."""

    def reply(request):
        if request.top_logprobs:
            return GenResponse(
                texts=["Sad"],
                first_token_logprobs=[
                    {"Sad": float(np.log(p_sad)), "Not": float(np.log(1 - p_sad))}
                ],
            )
        if request.thinking:
            return [f"<think>...</think>{think_answer}"] * request.n
        return ["Not sad"] * request.n

    return FunctionBackend(reply)


def test_confident_score_stops_at_the_first_rung(counter):
    spec, builder = _spec_builder(counter)
    backend = _scorer(0.97)
    conformal = SplitConformal(0.1).fit(np.array([[0.05, 0.95]] * 40), [1] * 40)
    prediction = Annotator(spec, builder, backend, conformal=conformal).annotate(
        Query("q", "tears")
    )
    assert (prediction.answer, prediction.source) == ("Sad", "score")
    assert backend.calls == 1


def test_unsure_score_escalates_to_think_and_vote(counter):
    spec, builder = _spec_builder(counter)
    backend = _scorer(0.55, think_answer="Not sad")
    prediction = Annotator(spec, builder, backend, StrategyConfig(conformal_alpha=None)).annotate(
        Query("q", "meh")
    )
    assert prediction.source == "think_vote" and prediction.answer == "Not sad"
    assert prediction.meta["votes"] == {"Not sad": 5.0}


def test_no_logprobs_falls_back_to_generation(counter):
    spec, builder = _spec_builder(counter)
    backend = FunctionBackend(lambda r: ["Sad"] * r.n)  # never returns probabilities
    prediction = Annotator(spec, builder, backend, StrategyConfig(escalate=False)).annotate(
        Query("q", "x")
    )
    assert prediction.answer == "Sad" and prediction.source == "generate"


def test_rejected_request_gets_a_model_free_answer(counter):
    spec, builder = _spec_builder(counter)

    def reject(request):
        raise BadRequestError("prompt too long")

    prediction = Annotator(spec, builder, FunctionBackend(reject)).annotate(
        Query("q", "lonely tears")
    )
    assert prediction.source == "fallback_knn" and prediction.answer in {"Sad", "Not sad"}
    assert prediction.confidence == 0.0


def test_dead_server_stops_the_run_instead_of_hiding_it(counter):
    spec, builder = _spec_builder(counter)

    def down(request):
        raise ServerUnavailableError("down")

    with pytest.raises(ServerUnavailableError):
        Annotator(spec, builder, FunctionBackend(down)).annotate(Query("q", "x"))


def test_invalid_answers_never_leave_the_annotator(counter):
    spec, builder = _spec_builder(counter)
    backend = FunctionBackend(lambda r: ["I cannot decide"] * r.n)
    prediction = Annotator(spec, builder, backend, StrategyConfig(use_scoring=False)).annotate(
        Query("q", "x")
    )
    assert prediction.answer in {"Sad", "Not sad"}


def test_verified_program_answers_need_no_model_call(counter):
    pool = [Example(f"e{i}", f"[{i}, {i + 3}]", "3") for i in range(400)]
    spec = get_task(1)
    builder = ContextBuilder(spec, pool, counter, CTX)
    backend = FunctionBackend(lambda r: "99")
    annotator = Annotator(spec, builder, backend, program_answers={"q": "3"})
    assert annotator.annotate(Query("q", "[1, 4]")).source == "program"
    assert backend.calls == 0
    other = annotator.annotate(Query("other", "[1, 9]"))  # no program answer: ladder continues
    assert other.answer == "99" and backend.calls > 0


GOOD_KERNEL = """```python
import torch
import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, o_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

def relu(input, inplace=False):
    out = torch.empty_like(input)
    _k[(triton.cdiv(input.numel(), 64),)](input, out, input.numel(), B=64)
    return out
```"""


def test_code_is_repaired_until_the_static_check_passes(counter):
    pool = [
        Example(f"k{i}", f"spec {i}", f"import triton\n# kernel {i}\n" + "x = 1\n" * 40)
        for i in range(60)
    ]
    spec = get_task(8)
    builder = ContextBuilder(spec, pool, counter, replace(CTX, tail_k=1))
    replies = iter(
        [[GOOD_KERNEL.replace("def relu(input, inplace=False)", "def my_relu(x)")], [GOOD_KERNEL]]
    )
    seen = []

    def reply(request):
        seen.append(request)
        return next(replies)

    strategy = StrategyConfig(codegen_candidates=1, codegen_execute=False)
    query = Query(
        "q",
        "Functional Description: relu\nWrapper Entry Information: relu(input, inplace=False) -> Tensor",
    )
    prediction = Annotator(spec, builder, FunctionBackend(reply), strategy).annotate(query)
    assert prediction.source == "codegen" and "def relu(input, inplace=False)" in prediction.answer
    assert "no top-level function named `relu`" in seen[1].messages[-1][1]
    assert "def relu(input, inplace=False):" in seen[0].messages[-1][1]  # the signature was pinned


def test_code_fallback_is_an_honest_stub(counter):
    pool = [Example(f"k{i}", f"spec {i}", "import triton\n" + "y = 2\n" * 40) for i in range(60)]
    spec = get_task(8)
    builder = ContextBuilder(spec, pool, counter, replace(CTX, tail_k=1))
    query = Query("q", "Wrapper Entry Information: tanh(input, *, out=None) -> Tensor")
    prediction = Annotator(spec, builder, FunctionBackend(lambda r: "no code, sorry")).annotate(
        query
    )
    assert (
        prediction.source == "fallback_stub"
        and "def tanh(input, *, out=None):" in prediction.answer
    )
    assert "NotImplementedError" in prediction.answer


# ---------------------------------------------------------------- journal


def test_journal_survives_a_torn_last_line(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.append(Prediction("a", "Sad", 0.9, "score"))
    journal.append(Prediction("b", "N", 0.5, "score"))
    with (tmp_path / "j.jsonl").open("a") as handle:
        handle.write('{"query_id": "c", "answ')  # the crash happened mid-write
    done = journal.load()
    assert set(done) == {"a", "b"} and done["a"].answer == "Sad"


def test_run_resumes_after_a_crash_without_repeating_work(counter, tmp_path):
    pool = make_sentiment_pool(600)
    task = TaskData(5, "toy", "sad?", tuple(pool), ())
    config = RunConfig(
        context=CTX, strategy=StrategyConfig(calibration="none", escalate=False), workers=2
    )
    queries = [Query(f"t{i}", f"lonely day {i}") for i in range(12)]
    calls = {"n": 0}
    lock = threading.Lock()

    def flaky(request):
        with lock:
            calls["n"] += 1
            if calls["n"] == 7:
                raise ServerUnavailableError("server died")
        return DemoReaderBackend().generate(request)

    journal = Journal(tmp_path / "run.jsonl")
    with pytest.raises(ServerUnavailableError):
        TaskRunner(task, FunctionBackend(flaky), counter, config).run(queries, journal)
    finished = len(journal.load())
    assert 0 < finished < 12
    backend = FunctionBackend(lambda r: DemoReaderBackend().generate(r))
    predictions = TaskRunner(task, backend, counter, config).run(queries, journal)
    assert [p.query_id for p in predictions] == [q.id for q in queries]
    assert backend.calls == 12 - finished  # only the unfinished inputs were asked again


# ----------------------------------------------------------------- memory


def test_memory_recalls_identical_inputs_and_gates_admission():
    memory = AnnotationMemory(max_items=10, min_confidence=0.9, max_label_share=0.6)
    query = Query("q1", "Same  Input")
    memory.record(query, Prediction("q1", "Sad", 0.99, "score"))
    recalled = memory.recall(Query("q2", "same input"))
    assert recalled.answer == "Sad" and recalled.source == "memory"
    assert not memory.record(
        Query("q3", "x"), Prediction("q3", "Sad", 0.5, "score")
    )  # not sure enough
    assert not memory.record(
        Query("q4", "y"), Prediction("q4", "Sad", 0.99, "fallback_knn")
    )  # untrusted
    assert memory.session() == ()  # nothing visible before the wave ends
    assert memory.commit() == 1 and len(memory.session()) == 1


def test_memory_keeps_one_label_from_taking_over():
    memory = AnnotationMemory(max_items=50, min_confidence=0.0, max_label_share=0.6)
    admitted = [
        memory.record(Query(f"q{i}", f"text {i}"), Prediction(f"q{i}", "Sad", 1.0, "score"))
        for i in range(10)
    ]
    assert admitted[:4] == [True] * 4 and not all(admitted)


def test_session_block_is_append_only_across_waves(counter):
    pool = make_sentiment_pool(600)
    task = TaskData(5, "toy", "sad?", tuple(pool), ())
    config = RunConfig(
        context=CTX,
        strategy=StrategyConfig(calibration="none", escalate=False),
        memory=True, memory_wave=4, memory_min_confidence=0.0, workers=1,
    )  # fmt: skip
    seen_sessions = []

    def reply(request):
        user = "".join(c for role, c in request.messages if role == "user")
        if "# Labelled earlier" in user:
            seen_sessions.append(
                user.split("# Labelled earlier in this session\n\n")[1].split("# Now label")[0]
            )
        return DemoReaderBackend().generate(request)

    runner = TaskRunner(task, FunctionBackend(reply), counter, config)
    runner.run([Query(f"t{i}", f"lonely rainy day number {i}") for i in range(12)])
    assert seen_sessions, "later waves should see earlier answers"
    for earlier, later in pairwise(seen_sessions):
        assert later.startswith(earlier)  # old entries never move, so the cached prefix stays valid


# -------------------------------------------------------------- scheduler


def test_scheduler_primes_each_group_then_fans_out_and_keeps_order():
    order = []
    lock = threading.Lock()

    def work(item):
        with lock:
            order.append(item)
        return item * 10

    items = [1, 2, 3, 4, 5, 6]
    results = run_grouped(items, key=lambda i: i % 2, work=work, workers=3)
    assert results == [10, 20, 30, 40, 50, 60]
    first_group = {i for i in items if i % 2 == order[0] % 2}
    assert set(order[: len(first_group)]) == first_group  # one group fully before the next


def test_scheduler_reraises_failures():
    def work(item):
        if item == 3:
            raise RuntimeError("boom")
        return item

    with pytest.raises(RuntimeError):
        run_grouped([1, 2, 3, 4], key=lambda i: 0, work=work, workers=2)


# ------------------------------------------------------------------ runner


def test_runner_prepares_calibration_and_conformal_from_left_over_examples(counter):
    pool = make_sentiment_pool(1500)
    task = TaskData(5, "toy", "sad?", tuple(pool), ())
    config = RunConfig(context=CTX, calib_limit=120, workers=4)
    runner = TaskRunner(task, DemoReaderBackend(noise=0.1), counter, config)
    report, _ = runner.evaluate("dev", limit=60)
    notes = runner.notes["calibration"]
    assert notes["method"] == "vector" and "conformal" in notes
    assert report.n == 60 and report.accuracy > 0.6
    assert set(report.by_source) <= {"score", "think_vote"}
    assert runner.meter.summary()["calibrate"]["calls"] == 120
    dev_ids = {e.id for e in runner.splits.dev}
    assert not dev_ids & {
        e.id for e in runner.builder.bulk_examples()
    }  # dev never leaks into prompts


def test_reports_are_json_serialisable(counter):
    pool = make_sentiment_pool(900)
    runner = TaskRunner(TaskData(5, "toy", "sad?", tuple(pool), ()), DemoReaderBackend(), counter,
                        RunConfig(context=CTX, calib_limit=60, workers=2))  # fmt: skip
    report, predictions = runner.evaluate("dev", limit=10)
    json.dumps({"eval": report.to_json(), "notes": runner.notes, "usage": runner.meter.summary(),
                "p": [p.to_json() for p in predictions]})  # fmt: skip


def test_runner_stack_mode_uses_signals_at_fit_and_predict_time(counter):
    pool = make_sentiment_pool(1500)
    task = TaskData(5, "toy", "sad?", tuple(pool), ())
    config = RunConfig(context=CTX, calib_limit=120, workers=2,
                       strategy=StrategyConfig(calibration="stack"))  # fmt: skip
    runner = TaskRunner(task, DemoReaderBackend(noise=0.3, seed=4), counter, config)
    report, _ = runner.evaluate("dev", limit=40)
    assert runner.notes["calibration"]["method"] == "stack"
    assert runner.signal_fn is not None and runner.signal_fn("lonely tears").shape == (2,)
    assert report.accuracy > 0.6


def test_untrusted_scores_send_every_input_to_think_and_vote(counter):
    """A one-label conformal set normally ends the ladder.  When held-out examples showed
    the sure answers are not accurate, the cheap rung is only a prior and everything
    climbs; the vote is then over all labels, not the cheap rung's (distrusted) set."""
    spec, builder = _spec_builder(counter)
    backend = _scorer(0.97, think_answer="Not sad")
    conformal = SplitConformal(0.1).fit(np.array([[0.05, 0.95]] * 40), [1] * 40)
    trusted = Annotator(spec, builder, backend, conformal=conformal).annotate(Query("q", "x"))
    assert trusted.source == "score" and trusted.answer == "Sad"
    doubted = Annotator(spec, builder, backend, conformal=conformal, trust_scores=False).annotate(
        Query("q", "x")
    )
    assert doubted.source == "think_vote" and doubted.answer == "Not sad"
    assert doubted.meta["plausible"] == ["Sad"]  # the cheap rung's opinion is still recorded


def test_runner_distrusts_a_cheap_rung_that_is_confidently_wrong(counter):
    """Held-out calibration examples catch a scorer that is sure and wrong.

    The fake scorer answers 'Sad' with 97% confidence for everything.  Two thirds of the
    synthetic pool is 'Not sad', so its sure answers are right about a third of the time.
    The runner must notice on the calibration half and stop trusting singletons."""
    pool = make_sentiment_pool(1500)
    task = TaskData(5, "toy", "sad?", tuple(pool), ())
    config = RunConfig(
        context=CTX,
        calib_limit=120,
        workers=2,
        # alpha 0.5 makes the conformal sets small, so the gate has sure answers to judge.
        strategy=StrategyConfig(calibration="vector", conformal_alpha=0.5),
    )
    runner = TaskRunner(task, _scorer(0.97, think_answer="Not sad"), counter, config)
    runner.prepare(runner.splits.as_queries("dev")[:10])
    conformal = runner.notes["calibration"]["conformal"]
    assert conformal["singletons_trusted"] is False
    assert conformal["singleton_count"] >= 10
    predictions = runner.run(runner.splits.as_queries("dev")[:6])
    assert {p.source for p in predictions} == {"think_vote"}


def test_program_induction_sees_a_short_example_block(counter):
    """The program-writing prompt gets a few dozen examples, not the 30K-token block."""
    pool = [Example(f"e{i}", f"[{i}, {i + 3}]", "3") for i in range(600)]
    task = TaskData(1, "gap", "smallest gap", tuple(pool), ())
    prompts: list[str] = []

    def reply(request):
        if request.tag == "induce_program":
            prompts.append(request.messages[-1][1])
            return "```python\ndef solve(text):\n    return '3'\n```"
        return "3"

    config = RunConfig(context=CTX, calib_limit=60, strategy=StrategyConfig(induction_examples=10))
    runner = TaskRunner(task, FunctionBackend(reply), counter, config)
    runner.prepare(runner.splits.as_queries("dev")[:5])
    assert runner.notes["program"]["accepted"] is True
    assert runner.notes["program"]["examples_shown"] == 10
    assert prompts and prompts[0].count("Input: ") == 10
    assert runner.notes["program"]["attempt_log"][0]["verify_accuracy"] == 1.0
