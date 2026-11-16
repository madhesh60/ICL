"""Finding the labelled examples that look most like the input we must label.

Everything here is plain counting and arithmetic: no second neural network.
That keeps the system inside a "one model only" rule, costs no GPU memory, and
gives the same result on every machine.
"""

from cairn.retrieval.bm25 import BM25Index
from cairn.retrieval.dense import HashedTfidf, spherical_kmeans
from cairn.retrieval.features import char_ngrams, lexical_features, word_tokens
from cairn.retrieval.fusion import mmr_select, reciprocal_rank_fusion

__all__ = [
    "BM25Index",
    "HashedTfidf",
    "char_ngrams",
    "lexical_features",
    "mmr_select",
    "reciprocal_rank_fusion",
    "spherical_kmeans",
    "word_tokens",
]
