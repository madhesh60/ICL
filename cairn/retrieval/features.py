"""Turning text into small comparable pieces ("features")."""

from __future__ import annotations

import re

# A word, a number, or any single symbol that is not a space.  Symbols are kept
# because in tweets an emoji such as a crying face carries most of the meaning.
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\sa-z\d]", re.IGNORECASE)


def word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    """Overlapping letter chunks.  They match "depress" inside "depressing"/"depressed"
    and survive typos, which matters for tweets and for names."""
    squeezed = re.sub(r"\s+", " ", text.lower()).strip()
    padded = f" {squeezed} "
    grams: list[str] = []
    for n in range(n_min, n_max + 1):
        grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


def lexical_features(text: str, *, bigrams: bool = True, chars: bool = False) -> list[str]:
    """Words, optional word pairs, optional letter chunks, each tagged with its kind."""
    words = word_tokens(text)
    features = [f"w:{w}" for w in words]
    if bigrams:
        features.extend(f"b:{a}_{b}" for a, b in zip(words, words[1:], strict=False))
    if chars:
        features.extend(f"c:{g}" for g in char_ngrams(text))
    return features
