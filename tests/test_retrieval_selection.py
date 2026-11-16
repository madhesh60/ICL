import subprocess
import sys

import numpy as np

from cairn.retrieval import (
    BM25Index,
    HashedTfidf,
    char_ngrams,
    lexical_features,
    mmr_select,
    reciprocal_rank_fusion,
    spherical_kmeans,
    word_tokens,
)
from cairn.selection import budgeted_coverage, coverage_value


def test_word_tokens_keep_emoji_and_numbers():
    assert word_tokens("I'm SO sad 😭 12") == ["i'm", "so", "sad", "😭", "12"]


def test_char_ngrams_match_word_variants():
    assert set(char_ngrams("depressing")) & set(char_ngrams("depressed"))


def test_bm25_prefers_rare_matching_words():
    docs = [word_tokens(t) for t in ["the cat sat", "the cat ran", "the tiramisu dessert", "the the the"]]
    index = BM25Index(docs)
    top = index.top_k(word_tokens("a tiramisu for the cat"), 2)
    assert top[0][0] == 2  # "tiramisu" (1 doc) beats "cat" (2 docs) beats "the" (all docs)
    assert index.idf("tiramisu") > index.idf("cat") > index.idf("the") > 0
    assert index.idf("never-seen") == 0.0


def test_bm25_mask_ties_and_empty_results():
    index = BM25Index([["a"], ["a"], ["b"]])
    assert [i for i, _ in index.top_k(["a"], 2)] == [0, 1]  # ties broken by index
    mask = np.array([False, True, True])
    assert [i for i, _ in index.top_k(["a"], 1, mask=mask)] == [1]
    assert index.top_k(["a"], 3, mask=np.zeros(3, dtype=bool)) == []


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 9]])
    assert fused[0][0] == 1  # near the top of both lists
    assert {item for item, _ in fused} == {1, 2, 3, 9}


def test_mmr_avoids_near_copies():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    picked = mmr_select([0, 1, 2], [1.0, 0.99, 0.5], vectors, k=2, trade_off=0.5)
    assert picked == [0, 2]  # the copy of item 0 is skipped
    assert mmr_select([0, 1, 2], [1.0, 0.99, 0.5], vectors, k=2, trade_off=1.0) == [0, 1]


def test_hashed_vectors_are_identical_in_another_process():
    """Python's own hash() changes between runs; ours must not, or clusters would move."""
    code = (
        "from cairn.retrieval import HashedTfidf;"
        "v = HashedTfidf(64).fit([['a','b'],['b','c']]).transform([['a','c']]);"
        "print(','.join(f'{x:.6f}' for x in v[0]))"
    )
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout for _ in range(2)}
    assert len(runs) == 1 and next(iter(runs)).strip()


def test_kmeans_separates_obvious_groups_and_is_repeatable():
    docs = [["cat", "pet"]] * 20 + [["stock", "market"]] * 20
    vectors = HashedTfidf(256).fit(docs).transform(docs)
    _, labels_a = spherical_kmeans(vectors, 2, seed=3)
    _, labels_b = spherical_kmeans(vectors, 2, seed=3)
    assert np.array_equal(labels_a, labels_b)
    assert len(set(labels_a[:20])) == 1 and len(set(labels_a[20:])) == 1 and labels_a[0] != labels_a[-1]


def _toy_problem(n=60, n_features=40, seed=0):
    rng = np.random.default_rng(seed)
    features = []
    for _ in range(n):
        ids = np.sort(rng.choice(n_features, size=rng.integers(2, 6), replace=False))
        features.append((ids, np.ones(len(ids))))
    weights = rng.random(n_features) + 0.1
    costs = [int(c) for c in rng.integers(5, 20, size=n)]
    return features, weights, costs


def test_coverage_reaches_budget_and_beats_random():
    features, weights, costs = _toy_problem()
    result = budgeted_coverage(features, weights, costs, min_total_cost=200)
    assert result.total_cost >= 200
    assert len(set(result.order)) == len(result.order)
    rng = np.random.default_rng(1)
    for _ in range(20):
        order, total = [], 0
        for i in rng.permutation(len(costs)):
            if total >= 200:
                break
            order.append(int(i))
            total += costs[i]
        # Same budget, random choice: never better than greedy on this objective.
        assert coverage_value(result.order, features, weights) >= coverage_value(order, features, weights) - 1e-9


def test_lazy_greedy_equals_plain_greedy():
    """The lazy shortcut must pick exactly what the slow, obvious algorithm picks."""
    features, weights, costs = _toy_problem(n=40, seed=5)
    lazy = budgeted_coverage(features, weights, costs, min_total_cost=150).order
    mass = np.zeros(len(weights))
    chosen, total, left = [], 0, set(range(len(costs)))
    while total < 150 and left:
        def ratio(i):
            ids, vals = features[i]
            return float(np.dot(weights[ids], np.sqrt(mass[ids] + vals) - np.sqrt(mass[ids]))) / costs[i]
        best = max(sorted(left), key=ratio)
        mass[features[best][0]] += features[best][1]
        chosen.append(best)
        total += costs[best]
        left.remove(best)
    assert lazy == chosen


def test_group_quotas_limit_a_dominant_group_but_never_break_the_budget():
    features, weights, costs = _toy_problem(n=80, seed=2)
    groups = ["big"] * 70 + ["small"] * 10
    result = budgeted_coverage(
        features, weights, costs, 300, groups=groups, group_share={"big": 0.5, "small": 0.5}
    )
    assert result.total_cost >= 300  # the rule wins over the preference
    small_cost = sum(costs[i] for i in result.order if groups[i] == "small")
    assert small_cost == sum(costs[70:])  # the small group was used up completely


def test_already_chosen_items_are_complemented_not_repeated():
    features = [(np.array([0]), np.ones(1)), (np.array([0]), np.ones(1)), (np.array([1]), np.ones(1))]
    result = budgeted_coverage(features, np.ones(2), [1, 1, 1], 1, already_chosen=[0])
    assert result.order == [2]  # feature 0 is covered already, so the new feature wins


def test_lexical_features_tag_kinds():
    feats = lexical_features("big cat", bigrams=True, chars=True)
    assert "w:big" in feats and "b:big_cat" in feats and any(f.startswith("c:") for f in feats)
