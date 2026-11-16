import numpy as np
import pytest

from cairn.decoding.calibrate import (
    BatchCalibrator,
    ContextualCalibrator,
    VectorScaling,
    expected_calibration_error,
)
from cairn.decoding.conformal import SplitConformal
from cairn.decoding.constraints import Choice, Regex, satisfies, to_request_fields
from cairn.decoding.parse import (
    extract_code,
    final_segment,
    make_label_normalizer,
    norm_int_list,
    norm_integer,
    norm_trivia,
    split_reasoning,
    strip_think,
)
from cairn.decoding.score import distinct_starts, options_from_logprobs
from cairn.decoding.vote import majority_vote

# ---------------------------------------------------------------- parsing


def test_strip_think_handles_finished_unfinished_and_half_blocks():
    assert strip_think("<think>hmm</think>\n\n42") == "42"
    assert strip_think("<think>still going and the budget ran o") == ""
    assert strip_think("half</think> 7") == "7"


def test_split_reasoning():
    assert split_reasoning("<think>a</think>b") == ("a", "b")
    assert split_reasoning("plain") == ("", "plain")


def test_final_segment_prefers_answer_marker_then_last_line():
    assert final_segment("blah\nAnswer: 12\nthanks") == "12"
    assert final_segment("first\n\nlast line") == "last line"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("33", "33"), ("The answer is 7.", "7"), ("Answer: -0", "0"), ("none", None)],
)
def test_norm_integer(text, expected):
    assert norm_integer(text) == expected


def test_norm_int_list_prints_like_python():
    assert norm_int_list("so the result is [36,88, 148]") == "[36, 88, 148]"
    assert norm_int_list("[1, 2.5]") is None
    assert norm_int_list("[True, 2]") is None  # bool is an int in Python; not here
    assert norm_int_list("no list") is None


def test_norm_trivia():
    assert norm_trivia("What is Tiramisu.") == "tiramisu"
    assert norm_trivia('"The Nutty Professor"') == "the nutty professor"


def test_label_normalizer_does_not_confuse_sad_with_not_sad():
    norm = make_label_normalizer(("Not sad", "Sad"))
    assert norm("not sad") == "Not sad"
    assert norm("The author is Not sad.") == "Not sad"
    assert norm("Sad") == "Sad"
    assert norm("saddle") is None


def test_extract_code_prefers_longest_block_and_survives_cut_off_fences():
    reply = "```python\nx = 1\n```\ntext\n```python\nimport triton\ndef f():\n    pass\n```"
    assert extract_code(reply).startswith("import triton")
    assert extract_code("```python\nimport torch\ndef g(): pass") == "import torch\ndef g(): pass"
    assert extract_code("no code here") is None


# ------------------------------------------------------------ constraints


def test_constraints_check_and_translate():
    choice = Choice(("Y", "N"))
    assert satisfies(choice, "Y") and not satisfies(choice, "y")
    assert satisfies(Regex(r"\[[0-9]+(, [0-9]+)*\]"), "[1, 2]")
    assert not satisfies(Regex(r"[0-9]+"), "12a")
    assert satisfies(None, "anything") and not satisfies(None, "  ")
    assert to_request_fields(choice, "structured_outputs") == {
        "structured_outputs": {"choice": ["Y", "N"]}
    }
    assert to_request_fields(Regex("a+"), "guided") == {"guided_regex": "a+"}
    assert to_request_fields(choice, "none") == {}
    with pytest.raises(ValueError):
        to_request_fields(choice, "bogus")


# ---------------------------------------------------------------- scoring


def test_distinct_starts():
    assert distinct_starts(["Sad", "Not sad"])
    assert distinct_starts(["Y", "N"])
    assert not distinct_starts(["no", "not sure"])  # "no" is a prefix of "not"


def test_options_from_logprobs_merges_pieces_and_floors_missing():
    logp = options_from_logprobs({"Sad": -0.2, " Not": -1.8, "The": -5.0}, ["Not sad", "Sad"])
    probs = np.exp(logp)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[1] > probs[0]
    missing = options_from_logprobs({"Y": -0.01}, ["N", "Y"])
    assert np.exp(missing[0]) > 0  # unlikely, never impossible


# ------------------------------------------------------------ calibration


def _biased_scores(n=600, seed=0):
    """True label decides a signal; the model adds a constant lean towards label 0."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    signal = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.0, size=n)
    logits = np.stack([-signal / 2 + 1.5, signal / 2], axis=1)  # +1.5 = the lean
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True)), y


def test_vector_scaling_removes_a_label_lean():
    logp, y = _biased_scores()
    before = (logp.argmax(1) == y).mean()
    cal = VectorScaling().fit(logp[:300], y[:300])
    after = (cal.transform(logp[300:]).argmax(1) == y[300:]).mean()
    assert after > before + 0.1


def test_batch_and_contextual_calibrators_reduce_lean():
    logp, y = _biased_scores()
    batch = BatchCalibrator().fit(logp)
    assert (batch.transform(logp).argmax(1) == y).mean() > (logp.argmax(1) == y).mean()
    contextual = ContextualCalibrator().fit(np.array([[np.log(0.9), np.log(0.1)]]))
    shifted = contextual.transform(np.array([[np.log(0.6), np.log(0.4)]]))
    assert shifted.argmax() == 1  # 0.6 vs a 0.9 lean is actually a vote for label 1
    with pytest.raises(RuntimeError):
        BatchCalibrator().transform(logp)


def test_ece_is_zero_when_confidence_matches_accuracy():
    assert expected_calibration_error([0.8] * 10, [True] * 8 + [False] * 2) == pytest.approx(0.0)
    assert expected_calibration_error([1.0] * 4, [False] * 4) == pytest.approx(1.0)


# --------------------------------------------------------------- conformal


def test_conformal_coverage_promise_holds_on_new_data():
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(30):
        y = rng.integers(0, 3, size=400)
        logits = rng.normal(0, 1, size=(400, 3))
        logits[np.arange(400), y] += 1.2
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        conformal = SplitConformal(alpha=0.1).fit(probs[:200], y[:200])
        covered.append(conformal.report(probs[200:], y[200:]).coverage)
    assert np.mean(covered) >= 0.88  # promised >= 0.90 on average; small tolerance for 30 trials


def test_conformal_sets_shrink_when_the_model_is_sure():
    sure = np.array([[0.98, 0.01, 0.01]] * 50)
    conformal = SplitConformal(0.1).fit(sure, [0] * 50)
    assert conformal.predict_sets(sure[:1]) == [[0]]
    unsure = np.array([[0.4, 0.35, 0.25]])
    assert len(conformal.predict_sets(unsure)[0]) <= 1
    with pytest.raises(ValueError):
        SplitConformal(alpha=1.5)


# -------------------------------------------------------------------- vote


def test_majority_vote_normalizes_before_counting():
    result = majority_vote(["Paris", "paris.", " PARIS ", "Rome"], norm_trivia)
    assert result.answer == "paris" and result.share == pytest.approx(0.75) and result.valid == 4


def test_majority_vote_ties_go_to_first_and_allowed_filters():
    assert majority_vote(["b", "a"], lambda x: x).answer == "b"
    result = majority_vote(["x", "x", "y"], lambda t: t, allowed=["y"])
    assert result.answer == "y" and result.valid == 1
    assert majority_vote(["", ""], lambda t: t or None).answer is None


def test_stacked_calibrator_learns_to_trust_a_useful_signal_and_ignore_noise():
    from cairn.decoding.calibrate import StackedCalibrator

    rng = np.random.default_rng(3)
    n = 800
    y = rng.integers(0, 2, size=n)
    weak_model = np.stack([rng.normal(0, 1, n), rng.normal(0, 1, n) + 0.3 * y], axis=1)
    useful = (y + rng.normal(0, 0.3, n))[:, None]  # a strong cheap signal
    noise = rng.normal(0, 1, (n, 3))  # three useless ones
    extra = np.hstack([useful, noise])
    stack = StackedCalibrator().fit(weak_model[:400], y[:400], extra[:400])
    accuracy = (stack.transform(weak_model[400:], extra[400:]).argmax(1) == y[400:]).mean()
    assert accuracy > 0.85
    weight_per_input = np.abs(stack.weights).sum(axis=1)
    assert weight_per_input[2] > 3 * weight_per_input[3:].max()  # signal >> noise
    with pytest.raises(ValueError):
        stack.transform(weak_model[:2], extra[:2, :2])


def test_every_calibrator_accepts_extra_signals():
    from cairn.decoding.calibrate import IdentityCalibrator

    logp, y = _biased_scores()
    extra = np.zeros((len(y), 2))
    for calibrator in (
        IdentityCalibrator(),
        BatchCalibrator().fit(logp),
        VectorScaling().fit(logp, y),
    ):
        assert calibrator.transform(logp, extra).shape == logp.shape
