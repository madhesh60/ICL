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
    # Examples are executable asserts with repr(): the input's type is unambiguous.
    assert prompts and prompts[0].count("assert solve('[") == 10
    assert runner.notes["program"]["attempt_log"][0]["verify_accuracy"] == 1.0


# ------------------------------------------------- escalation probe (E001 follow-up)


def _trivia_task(n: int = 1500) -> TaskData:
    pool = [
        Example(f"e{i}", f"Category: TOPIC{i % 7} \nClue: clue number {i}", f"answer{i}")
        for i in range(n)
    ]
    return TaskData(7, "toy trivia", "answer the clue", tuple(pool), ())


def _knower(thinking_knows: bool, direct_knows: bool):
    """A fake model that reads the clue number from the prompt and may or may not know it."""
    import re

    def reply(request):
        last = request.messages[-1][1]
        number = re.findall(r"clue number (\d+)", last)[-1]
        knows = thinking_knows if request.thinking else direct_knows
        answer = f"answer{number}" if knows else "no idea"
        return [f"<think>...</think>{answer}" if request.thinking else answer] * request.n

    return FunctionBackend(reply)


def test_probe_picks_reason_first_when_the_cheap_rung_is_confidently_wrong(counter):
    """E001 task 2 in miniature: the cheap samples agree, and they are wrong.

    Agreement is a measure of confidence, not of correctness.  Only a paired comparison on
    held-out examples shows that thinking is needed even where the cheap rung is sure."""
    config = RunConfig(
        context=CTX, calib_limit=80, workers=2, strategy=StrategyConfig(escalation_probe=30)
    )
    runner = TaskRunner(_trivia_task(), _knower(True, False), counter, config)
    dev = runner.splits.as_queries("dev")[:6]
    predictions = runner.run(dev)
    policy = runner.notes["policy"]
    assert policy["chosen"] == "reason_first"
    assert policy["sure"]["n"] == 30 and policy["sure"]["cheap_accuracy"] == 0.0
    assert policy["sure"]["costly_accuracy"] == 1.0 and policy["sure"]["p_value"] < 0.001
    assert {p.source for p in predictions} == {"think_vote"}
    assert all(p.answer == q.gold for p, q in zip(predictions, dev, strict=True))


def test_probe_picks_cheap_only_when_thinking_does_not_help(counter):
    """E001 task 7 in miniature: both rungs equally (un)informed, so do not pay for thinking."""
    config = RunConfig(
        context=CTX, calib_limit=80, workers=2, strategy=StrategyConfig(escalation_probe=30)
    )
    runner = TaskRunner(_trivia_task(), _knower(True, True), counter, config)
    runner.prepare(runner.splits.as_queries("dev")[:4])
    assert runner.notes["policy"]["chosen"] == "cheap_only"
    assert runner.config.strategy.escalate is False
    calls_before = runner.meter.summary().get("think_vote", {}).get("calls", 0)
    runner.run(runner.splits.as_queries("dev")[:4])
    assert runner.meter.summary()["think_vote"]["calls"] == calls_before  # no new thinking


def test_reason_first_routes_a_classification_task_straight_to_thinking(counter):
    """The test that would have caught the E001 task-2 miss: `use_scoring: false` alone does
    NOT reach the thinking rung (the cheap constrained samples answer first)."""
    spec, builder = _spec_builder(counter)

    def reply(request):
        return [
            f"<think>listing words</think>{'Sad' if request.thinking else 'Not sad'}"
        ] * request.n

    plain = Annotator(spec, builder, FunctionBackend(reply), StrategyConfig(use_scoring=False))
    assert plain.annotate(Query("q", "x")).source == "generate"  # the E001 behaviour
    first = Annotator(
        spec, builder, FunctionBackend(reply), StrategyConfig(use_scoring=False, reason_first=True)
    )
    prediction = first.annotate(Query("q", "x"))
    assert (prediction.source, prediction.answer) == ("think_vote", "Sad")


def test_a_kernel_proven_wrong_is_replaced_by_a_loud_stub(counter, monkeypatch):
    """S_exec^2 / S_call: handing in a kernel known to be wrong can only lower the score."""
    from cairn.pipeline import annotator as annotator_module
    from cairn.tasks.kernel_exec import ExecReport, consistency_score

    monkeypatch.setattr(
        annotator_module,
        "execute_kernel",
        lambda code, signature: ExecReport("wrong_result", "differs from torch.relu"),
    )
    pool = [Example(f"k{i}", f"spec {i}", "import triton\n" + "y = 2\n" * 40) for i in range(60)]
    spec = get_task(8)
    builder = ContextBuilder(spec, pool, counter, replace(CTX, tail_k=1))
    query = Query("q", "Wrapper Entry Information: relu(input, inplace=False) -> Tensor")
    backend = FunctionBackend(lambda r: [GOOD_KERNEL] * r.n)
    strategy = StrategyConfig(codegen_candidates=1, codegen_rounds=2)
    prediction = Annotator(spec, builder, backend, strategy).annotate(query)
    assert prediction.source == "abstain" and "NotImplementedError" in prediction.answer
    assert prediction.meta["exec_status"] == "abstained"
    assert "def relu" in prediction.meta["abstained_code"]  # kept for analysis
    kept = Annotator(
        spec, builder, backend, replace(strategy, abstain_on_proven_wrong=False)
    ).annotate(query)
    assert kept.source == "codegen" and kept.meta["exec_status"] == "wrong_result"
    # 10 right, 30 callable, out of 166 -> abstaining on the 20 proven wrong triples the score.
    assert consistency_score(10, 10, 166) > 2.9 * consistency_score(30, 10, 166)


def test_probe_picks_the_ladder_when_thinking_helps_only_where_the_cheap_rung_is_unsure(counter):
    """Even clue numbers: the quick answers agree and are right.  Odd ones: the quick
    samples scatter (the cheap rung is unsure) and only thinking knows the answer."""
    import re

    def reply(request):
        number = int(re.findall(r"clue number (\d+)", request.messages[-1][1])[-1])
        if request.thinking:
            return [f"<think>...</think>answer{number}"] * request.n
        if number % 2 == 0:
            return [f"answer{number}"] * request.n
        if request.temperature > 0:
            return [f"guess {i}" for i in range(request.n)]
        return ["guess x"] * request.n

    config = RunConfig(
        context=CTX, calib_limit=80, workers=2, strategy=StrategyConfig(escalation_probe=30)
    )
    runner = TaskRunner(_trivia_task(), FunctionBackend(reply), counter, config)
    runner.prepare(runner.splits.as_queries("dev")[:4])
    policy = runner.notes["policy"]
    assert policy["chosen"] == "ladder", policy
    assert policy["sure"]["only_costly_right"] == 0 and policy["unsure"]["only_costly_right"] > 5
    assert runner.config.strategy.escalate is True and runner.config.strategy.reason_first is False


# ------------------------------------------------- E002 follow-up: listing rung, probe rule


def _count_spec_builder(counter):
    pool = [
        Example(
            f"c{i}",
            f"Sentence: 'a dog number {i} sits on a mat'. Count the number of nouns in this sentence.",
            "3",
        )
        for i in range(400)
    ]
    spec = get_task(2, tuple("0123456"))
    return spec, ContextBuilder(spec, pool, counter, CTX, definition="count")


COUNT_Q = Query(
    "q",
    "Sentence: 'A man riding a white horse on a sandy beach'. Count the number of nouns in this sentence.",
)


def test_the_listing_rung_counts_in_code(counter):
    """The model names the words; len() does the counting.  Votes are over counts."""
    spec, builder = _count_spec_builder(counter)
    seen = []

    def reply(request):
        seen.append(request)
        if request.tag == "decompose":
            return ["Nouns: man, horse, beach"] * (request.n - 1) + ["Nouns: man, beach"]
        return ["9"] * request.n

    prediction = Annotator(
        spec, builder, FunctionBackend(reply), StrategyConfig(use_scoring=False)
    ).annotate(COUNT_Q)
    assert (prediction.answer, prediction.source) == ("3", "decompose")
    assert prediction.meta["votes"] == {"3": 4.0, "2": 1.0}
    request = seen[0]
    assert request.thinking is False and request.constraint is not None
    assert "list the nouns" in request.messages[-1][1]
    assert "tennis court" in request.messages[0][1]  # the counting rules are in the system prompt


def test_a_rejected_format_constraint_costs_a_retry_not_the_input(counter):
    spec, builder = _count_spec_builder(counter)
    constraints = []

    def reply(request):
        if request.tag == "decompose":
            constraints.append(request.constraint)
            if request.constraint is not None:
                raise BadRequestError("regex not supported")
            return ["Nouns: man, horse, beach"] * request.n
        return ["9"] * request.n

    prediction = Annotator(
        spec, builder, FunctionBackend(reply), StrategyConfig(use_scoring=False)
    ).annotate(COUNT_Q)
    assert prediction.source == "decompose" and prediction.answer == "3"
    assert constraints[0] is not None and constraints[1] is None


def test_unreadable_lists_hand_over_to_the_normal_ladder(counter):
    spec, builder = _count_spec_builder(counter)

    def reply(request):
        if request.tag == "decompose":
            return ["I am not sure"] * request.n
        return ["2"] * request.n

    strategy = StrategyConfig(use_scoring=False, escalate=False)
    prediction = Annotator(spec, builder, FunctionBackend(reply), strategy).annotate(COUNT_Q)
    assert (prediction.answer, prediction.source) == ("2", "generate")


def test_empty_thinking_vote_recovers_with_exact_length_word_mask(counter):
    spec, builder = _count_spec_builder(counter)
    requests = []

    def reply(request):
        requests.append(request)
        if request.thinking:
            return [""] * request.n
        if request.constraint is not None:
            raise BadRequestError("no regex support")
        return ["0100010001", "0100010001", "1", "01oops10001", ""]

    # Ten word positions, including two separate occurrences of the article.
    query = Query(
        "q",
        "Sentence: 'A man walking with a dog on the sandy beach'. Count the number of nouns in this sentence.",
    )
    cfg = StrategyConfig(use_scoring=False, reason_first=True, use_word_mask=True)
    p = Annotator(spec, builder, FunctionBackend(reply), cfg).annotate(query)
    assert p.source == "decompose" and p.answer == "3"
    assert p.confidence == 2 / 5
    assert p.meta["votes"] == {"3": 2.0} and p.meta["valid_think_votes"] == 0
    assert [r.tag for r in requests] == ["think_vote", "decompose", "decompose"]
    assert "single digit" not in requests[1].messages[0][1]


def test_unsure_decomposition_reaches_measured_thinking_rung(counter):
    spec, builder = _count_spec_builder(counter)
    backend = FunctionBackend(
        lambda r: (
            ["Nouns: man", "Nouns: man, horse", "Nouns: beach", "", ""]
            if r.tag == "decompose"
            else ["3", "", "", "", ""]
        )
    )
    p = Annotator(spec, builder, backend, StrategyConfig(use_scoring=False)).annotate(COUNT_Q)
    assert p.source == "think_vote" and p.answer == "3" and p.confidence == 0.2


def test_clue_expansion_changes_search_without_rewriting_the_question(counter):
    runner = TaskRunner(
        _trivia_task(),
        FunctionBackend(
            lambda r: ["river tributary"] if r.tag == "clue_expansion" else ["river"] * r.n
        ),
        counter,
        RunConfig(context=CTX, strategy=StrategyConfig(reason_first=True, use_clue_expansion=True)),
    )
    q = runner.splits.as_queries("dev")[0]
    runner.prepare([q])
    pred = runner.run([q])[0]
    assert runner.meter.summary()["clue_expansion"]["calls"] == 1
    assert pred.answer == "river" and "new_tail_examples" in pred.meta
    revised = runner.builder.build(q, tail_query=q.input + " river tributary")
    assert (
        q.input in revised.messages[-1]["content"]
        and "river tributary" not in revised.messages[-1]["content"]
    )
    assert revised.prefix_key == runner.builder.build(q).prefix_key


def test_probe_acts_on_an_overall_difference_that_each_group_alone_cannot_show(counter):
    """E002 task 7: thinking won 9-2 overall (p = 0.065) but no group alone was significant.

    Here 10 probe inputs: 5 the cheap rung is sure about and wrong, 5 it is unsure about and
    wrong; thinking is right on all 10.  Each group is 5-0, p = 0.0625 > 0.05; overall 10-0,
    p = 0.002.  The old rule chose cheap_only; the new one must not."""
    import re

    def reply(request):
        number = int(re.findall(r"clue number (\d+)", request.messages[-1][1])[-1])
        if request.thinking:
            return [f"<think>...</think>answer{number}"] * request.n
        if number % 2 == 0 or request.temperature == 0:
            return ["wrong"] * request.n
        return [f"guess {i}" for i in range(request.n)]

    config = RunConfig(
        context=CTX,
        calib_limit=80,
        workers=2,
        strategy=StrategyConfig(escalation_probe=10, probe_p_value=0.05),
    )
    runner = TaskRunner(_trivia_task(), FunctionBackend(reply), counter, config)
    runner.prepare(runner.splits.as_queries("dev")[:2])
    policy = runner.notes["policy"]
    assert policy["all"]["p_value"] < 0.01
    assert policy["sure"]["p_value"] > 0.05 or policy["unsure"]["p_value"] > 0.05
    assert policy["chosen"] in {"reason_first", "ladder"}, policy
