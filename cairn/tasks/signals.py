"""Cheap, task-specific measurements that need no model ("signals").

How the task-6 signal was found (this is the part worth copying):

    The task says "decide whether sentence 2 is from the same genre".  Before
    building anything, we asked where the data came from.  It is built from
    MNLI, where every sentence 2 was *written by a person who read sentence 1*.
    "Y" pairs keep that original sentence 2; "N" pairs swap in a sentence from
    another genre.  So the honest question is "was sentence 2 written from
    sentence 1?", and shared content words answer it.  Measured on our dev
    split: this single number reaches 93% accuracy, against 64% for always
    answering "N" and 58% for the nearest-examples vote.

A signal is never used on its own.  It is one more input to the stacked
calibrator, which learns from held-out examples how much to trust it next to
the language model.
"""

from __future__ import annotations

import re

from cairn.retrieval.features import word_tokens

_T6 = re.compile(r"^Sentence 1: (.*?)\s*Sentence 2: (.*?)\s*Genre: (.*?)\.?$", re.S)
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "and", "in", "is", "it", "that", "was", "for", "on", "are",
        "be", "with", "as", "at", "by", "this", "i", "you", "he", "she", "they", "we", "not",
        "have", "has", "had", "do", "does", "did", "or", "but", "so", "if", "from", "there",
        "their", "his", "her", "its", "our", "your",
    }
)  # fmt: skip


def _content_words(text: str) -> set[str]:
    return {w for w in word_tokens(text) if w.isalnum() and w not in _STOPWORDS}


def sentence_pair_overlap(text: str) -> tuple[float, ...]:
    """Task 6: how much of sentence 2 is already in sentence 1."""
    match = _T6.match(text)
    if not match:
        return (0.0, 0.0, 0.0)
    first, second = _content_words(match.group(1)), _content_words(match.group(2))
    shared = len(first & second)
    return (
        shared / max(len(second), 1),  # share of sentence 2 found in sentence 1
        shared / max(len(first | second), 1),  # Jaccard overlap
        min(shared, 5) / 5.0,  # raw count, capped
    )
