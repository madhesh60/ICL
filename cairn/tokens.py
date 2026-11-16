"""Counting tokens the way Qwen3 counts them.

The contest rule is written in tokens ("at least 30K tokens of examples"), and the
model's memory limit is in tokens too.  So we need a ruler.  Three rulers, best first:

    1. HFTokenCounter      - the real tokenizer from the model folder.  Exact.
    2. TiktokenQwenCounter - the Qwen vocabulary file that ships in this repo, run
                             through `tiktoken`.  Qwen3 uses the same 151,643 word
                             pieces, so counts match on normal text.
    3. HeuristicCounter    - no files needed, a rough estimate.

A ruler that is not exact tells us how much it may be off (`margin`).  The context
builder adds that margin on top of the minimum, so an inexact ruler can make the
prompt a bit longer than needed but can never make it break the rule.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from cairn.data import REPO_ROOT

log = logging.getLogger(__name__)

# The split pattern Qwen uses before byte-pair merging.  Note `\p{N}`: a number is
# cut into single digits, so "1234" costs 4 tokens.
_QWEN_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]+|\s+(?!\S)|\s+"""
)
_QWEN_SPECIAL = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<think>": 151667,
    "</think>": 151668,
}
DEFAULT_VOCAB = REPO_ROOT / "OpenSeek" / "hf_openseek" / "tokenizer" / "qwen.tiktoken"


@runtime_checkable
class TokenCounter(Protocol):
    name: str
    margin: float  # how far off the count may be, as a fraction (0.0 = exact)

    def encode(self, text: str) -> list[int]: ...

    def count(self, text: str) -> int: ...


class _CachedCount:
    """Mixin: remember counts, because the same example is measured many times."""

    _cache: dict[str, int]

    def count(self, text: str) -> int:
        hit = self._cache.get(text)
        if hit is None:
            hit = len(self.encode(text))  # type: ignore[attr-defined]
            if len(self._cache) < 500_000:
                self._cache[text] = hit
        return hit


class TiktokenQwenCounter(_CachedCount):
    name = "tiktoken-qwen"
    margin = 0.01

    def __init__(self, vocab_path: Path | str = DEFAULT_VOCAB) -> None:
        import tiktoken  # optional dependency, imported only when this ruler is chosen

        path = Path(vocab_path)
        with path.open("rb") as handle:
            ranks = {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split() for line in handle.read().splitlines() if line)
            }
        self._enc = tiktoken.Encoding(
            "qwen", pat_str=_QWEN_PATTERN, mergeable_ranks=ranks, special_tokens=_QWEN_SPECIAL
        )
        self._cache = {}

    def encode(self, text: str) -> list[int]:
        # Qwen normalises text to NFC first ("e" + accent mark becomes one letter).
        return self._enc.encode(unicodedata.normalize("NFC", text), allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


class HFTokenCounter(_CachedCount):
    name = "hf-tokenizer"
    margin = 0.0

    def __init__(self, model_path: Path | str) -> None:
        from transformers import AutoTokenizer  # optional dependency

        self._tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self._cache = {}

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


class HeuristicCounter(_CachedCount):
    """A ruler that needs no files.  Rough, so it carries a large safety margin."""

    name = "heuristic"
    margin = 0.15
    _piece = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]|\s+")

    def __init__(self) -> None:
        self._cache = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._piece.findall(text):
            if piece.isalpha():
                # An English word is about one token per 4-5 letters.
                ids.extend([0] * max(1, math.ceil(len(piece) / 4.5)))
            elif piece.isspace():
                # A single space is glued to the next word; line breaks cost a token.
                if "\n" in piece:
                    ids.append(1)
            else:
                ids.append(2)
        return ids


def get_counter(tokenizer_path: Path | str | None = None) -> TokenCounter:
    """Pick the best ruler that works on this machine.

    Order: an explicit model folder, then `CAIRN_TOKENIZER`, then the bundled Qwen
    vocabulary, then the heuristic.  We log which one was chosen because a silent
    fallback to a worse ruler is exactly the kind of thing that hides bugs.
    """
    path = tokenizer_path or os.environ.get("CAIRN_TOKENIZER")
    if path:
        try:
            counter: TokenCounter = HFTokenCounter(path)
            log.info("token counter: exact Hugging Face tokenizer at %s", path)
            return counter
        except Exception as err:  # missing package, missing files, bad folder
            log.warning("could not load tokenizer at %s (%s); trying the bundled vocabulary", path, err)
    vocab = Path(os.environ.get("CAIRN_QWEN_VOCAB", DEFAULT_VOCAB))
    try:
        counter = TiktokenQwenCounter(vocab)
        log.info("token counter: tiktoken with the bundled Qwen vocabulary")
        return counter
    except Exception as err:
        log.warning(
            "could not build the tiktoken counter (%s); using the rough heuristic with a %d%% "
            "safety margin. Install `tiktoken` for near-exact counts.",
            err,
            int(HeuristicCounter.margin * 100),
        )
        return HeuristicCounter()


def common_prefix_len(a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray) -> int:
    """How many tokens two prompts share from the start (what a KV cache can reuse)."""
    limit = min(len(a), len(b))
    if limit == 0:
        return 0
    differs = np.asarray(a[:limit]) != np.asarray(b[:limit])
    first = int(np.argmax(differs))
    return limit if not differs[first] else first
