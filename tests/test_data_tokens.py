import pytest

from cairn.context.render import render_demo
from cairn.data import DataError, SplitSizes, load_task, split_examples
from cairn.tasks import get_task
from cairn.tokens import HeuristicCounter, common_prefix_len, get_counter
from cairn.types import Example


def _examples(n: int) -> list[Example]:
    return [Example(f"id{i}", f"input {i}", str(i % 3)) for i in range(n)]


def test_split_roles_never_overlap():
    splits = split_examples(_examples(3000))
    roles = [splits.pool, splits.calib, splits.dev, splits.holdout]
    ids = [e.id for role in roles for e in role]
    assert len(ids) == len(set(ids)) == 3000
    assert (len(splits.calib), len(splits.dev), len(splits.holdout)) == (300, 300, 200)


def test_split_is_stable_under_reordering_and_growth():
    examples = _examples(2000)
    first = split_examples(examples)
    shuffled = split_examples(list(reversed(examples)))
    assert {e.id for e in first.dev} == {e.id for e in shuffled.dev}
    assert split_examples(examples, seed=1).dev != first.dev


def test_small_task_scales_reserved_sizes_down():
    splits = split_examples(_examples(100), SplitSizes(max_reserved_frac=0.35))
    reserved = len(splits.calib) + len(splits.dev) + len(splits.holdout)
    assert reserved <= 35 and len(splits.pool) >= 65


def test_pool_keeps_file_order():
    splits = split_examples(_examples(500))
    numbers = [int(e.id[2:]) for e in splits.pool]
    assert numbers == sorted(numbers)


def test_load_task_rejects_bad_id_and_missing_folder(tmp_path):
    with pytest.raises(DataError):
        load_task(9)
    with pytest.raises(DataError, match="CAIRN_DATA_DIR"):
        load_task(1, tmp_path)


@pytest.mark.data
def test_all_tasks_load_with_unique_ids(task_loader):
    sizes = {1: (5500, 500), 5: (1899, 500), 8: (184, 166)}
    for task_id, (n_examples, n_test) in sizes.items():
        task = task_loader(task_id)
        assert (len(task.examples), len(task.test)) == (n_examples, n_test)
        assert not {e.id for e in task.examples} & {q.id for q in task.test}


def test_common_prefix_len():
    assert common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([5], [6]) == 0


def test_heuristic_counter_has_a_safety_margin():
    counter = HeuristicCounter()
    assert counter.margin > 0.1
    assert counter.count("hello world 123") >= 5  # 2 words + 3 digits


@pytest.mark.data
def test_token_counts_add_up_across_examples(counter, task_loader):
    """The budget is computed by adding per-example counts.  That is only valid if
    joining examples never merges tokens across the boundary.  Checked on real data."""
    if counter.name == "heuristic":
        pytest.skip("needs a real tokenizer")
    for task_id in (1, 5, 7, 8):
        task = task_loader(task_id)
        spec = get_task(task_id)
        rendered = [render_demo(spec, e) for e in task.examples[:60]]
        assert counter.count("".join(rendered)) == sum(counter.count(r) for r in rendered)


def test_numbers_cost_one_token_per_digit(counter):
    if counter.name == "heuristic":
        pytest.skip("needs a real tokenizer")
    assert counter.count("12345") == 5


def test_get_counter_falls_back_when_path_is_wrong(tmp_path):
    counter = get_counter(tmp_path / "no-such-model")
    assert counter.name in {"tiktoken-qwen", "heuristic"}
