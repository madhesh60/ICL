from dataclasses import replace

import pytest

from cairn.context import ContextBuilder, ContextConfig
from cairn.context.render import render_demo, system_prompt
from cairn.data import split_examples
from cairn.eval.contextbench import context_report, render_chat, simulate_prefix_cache
from cairn.tasks import get_task
from cairn.types import Example, Query
from tests.conftest import make_sentiment_pool

SMALL = ContextConfig(min_demo_tokens=800, tail_k=4, tail_per_label=1)


def _builder(counter, config=SMALL, pool=None, **kwargs) -> ContextBuilder:
    pool = pool or make_sentiment_pool()
    spec = get_task(5, tuple(e.output for e in pool))
    return ContextBuilder(spec, pool, counter, config, definition="Is the author sad?", **kwargs)


def test_every_prompt_meets_the_minimum_from_shared_blocks_alone(counter):
    builder = _builder(counter)
    for i in range(20):
        prompt = builder.build(Query(f"q{i}", f"so lonely and tears today {i}"))
        assert prompt.stats.bulk_tokens >= 800  # the rule holds even without the tail
        assert prompt.stats.meets_minimum


def test_inexact_counter_gets_a_safety_margin(heuristic_counter):
    builder = _builder(heuristic_counter)
    prompt = builder.build(Query("q", "a fun sunny day"))
    assert prompt.stats.bulk_tokens >= 800 * (1 + heuristic_counter.margin)


def test_pool_too_small_is_a_clear_error(counter):
    with pytest.raises(ValueError, match="required"):
        _builder(counter, replace(SMALL, min_demo_tokens=10**7))
    with pytest.raises(ValueError, match="empty"):
        ContextBuilder(get_task(5), [], counter)


def test_prompts_share_their_beginning(counter):
    builder = _builder(counter)
    a = builder.build(Query("a", "awful grief"))
    b = builder.build(Query("b", "great fun"))
    assert a.prefix_key == b.prefix_key
    assert a.messages[0] == b.messages[0]
    assert (
        a.messages[1]["content"].split("# Now label")[0]
        == b.messages[1]["content"].split("# Now label")[0]
    )


def test_tail_has_nearest_last_and_every_label(counter):
    builder = _builder(counter)
    prompt = builder.build(Query("q", "lonely tears grief hurt"))
    turns = [m for m in prompt.messages[1:] if m["role"] == "assistant"]
    assert {t["content"] for t in turns} == {"Sad", "Not sad"}
    last_input = prompt.messages[-3]["content"].rsplit("Input: ", 1)[1]
    first_ranked = builder.nearest("lonely tears grief hurt", 1)[0][0]
    assert (
        last_input == builder.pool[first_ranked].input
    )  # the best match sits right before the query
    assert prompt.messages[-1]["content"].endswith("Input: lonely tears grief hurt")


def test_tail_as_text_mode_and_rule_reminder(counter):
    builder = _builder(counter, replace(SMALL, tail_as_turns=False))
    prompt = builder.build(Query("q", "a sunny win"))
    assert len(prompt.messages) == 2
    content = prompt.messages[1]["content"]
    assert content.endswith("Input: a sunny win\nOutput:")
    assert '"Sad" or "Not sad"' in content.split("# Your turn")[1]


def test_bulk_order_breaks_label_streaks(counter):
    pool = [Example(f"a{i}", f"happy text {i}", "Not sad") for i in range(200)]
    pool += [Example(f"b{i}", f"sad text {i}", "Sad") for i in range(200)]
    first = _builder(counter, replace(SMALL, selection="first"), pool=pool)
    mixed = _builder(counter, replace(SMALL, bulk_order="interleave"), pool=pool)
    queries = [Query("q", "text")]
    assert context_report(mixed, queries).longest_label_streak <= 3
    assert context_report(first, queries).longest_label_streak > 20


def test_label_balance_moves_shares_towards_equal(counter):
    natural = _builder(counter, replace(SMALL, label_balance=0.0))
    equal = _builder(counter, replace(SMALL, label_balance=1.0))

    def sad_share(b):
        labels = [e.output for e in b.bulk_examples()]
        return labels.count("Sad") / len(labels)

    assert sad_share(equal) > sad_share(natural)


def test_clusters_route_consistently_and_keep_the_minimum(counter):
    builder = _builder(counter, replace(SMALL, n_clusters=3, global_frac=0.5))
    assert builder.n_clusters == 3
    for text in ["lonely tears", "sunny fun win", "the day"]:
        prompt = builder.build(Query("q", text))
        assert prompt.cluster == builder.route(text)
        assert prompt.stats.bulk_tokens >= 800


def test_session_block_goes_after_the_bulk_and_before_the_tail(counter):
    builder = _builder(counter)
    session = [Example("s1", "remembered input", "Sad")]
    content = builder.build(Query("q", "lonely"), session).messages[1]["content"]
    assert (
        content.index("# Reference examples")
        < content.index("# Labelled earlier")
        < content.index("# Now label")
    )


def test_with_guidelines_changes_only_the_system_prompt(counter):
    builder = _builder(counter)
    ruled = builder.with_guidelines("- emojis alone do not decide")
    assert "emojis alone" in ruled.build(Query("q", "x")).messages[0]["content"]
    assert ruled.bulk_examples() == builder.bulk_examples()
    assert "emojis alone" not in builder.build(Query("q", "x")).messages[0]["content"]


def test_system_prompt_has_no_invented_example_labels():
    text = system_prompt(get_task(1), "Find the smallest gap.")
    assert "Good Review" not in text and "<label>" not in text


def test_render_code_demo_is_fenced():
    demo = render_demo(get_task(8), Example("e", "spec", "import triton\n"))
    assert "```python\nimport triton\n```" in demo and demo.endswith("\n\n")


def test_cache_simulation_rewards_grouping(counter):
    builder = _builder(counter, replace(SMALL, n_clusters=3, global_frac=0.0))
    queries = [
        Query(f"q{i}", t) for i, t in enumerate(["lonely tears", "sunny fun", "the day"] * 6)
    ]
    grouped = simulate_prefix_cache(builder, queries, counter, grouped=True, slots=1)
    arrival = simulate_prefix_cache(builder, queries, counter, grouped=False, slots=1)
    assert grouped.cache_share > arrival.cache_share
    assert render_chat([{"role": "user", "content": "x"}]) == "<|im_start|>user\nx<|im_end|>\n"


@pytest.mark.data
def test_real_tasks_meet_their_official_minimum(counter, task_loader):
    for task_id in (2, 6, 8):
        task = task_loader(task_id)
        splits = split_examples(task.examples)
        spec = get_task(task_id, tuple(e.output for e in splits.pool))
        builder = ContextBuilder(
            spec, splits.pool, counter, ContextConfig(tail_k=2), definition=task.definition
        )
        prompt = builder.build(task.test[0])
        assert prompt.stats.bulk_tokens >= spec.min_context_tokens
        # The counted prompt is close to a fresh count of the real text.
        real = counter.count("".join(m["content"] for m in prompt.messages))
        assert abs(prompt.stats.prompt_tokens - real) / real < 0.02
