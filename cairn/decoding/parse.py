"""Turning raw model text into a clean answer.

The official baseline lost 5% to 61% of its answers here: the model wrote a fine
answer, the parser could not find it, and the sample was submitted as `null`.
The parser below never returns "nothing" silently.  It returns a cleaned answer
or `None`, and the pipeline has a fallback for `None`.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_MARK = re.compile(r"(?:final answer|answer|output|label)\s*[:=]\s*", re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_INT = re.compile(r"-?\d+")
_QUESTION_LEAD = re.compile(r"^(?:what|who|where|when)\s+(?:is|are|was|were)\s+", re.IGNORECASE)

Normalizer = Callable[[str], str | None]


def strip_think(text: str) -> str:
    """Remove the model's private reasoning, including an unfinished block."""
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:  # the token budget ran out while still thinking
        return ""
    if "</think>" in text:  # the server already cut the opening tag
        text = text.split("</think>", 1)[1]
    return text.strip()


def split_reasoning(text: str) -> tuple[str, str]:
    """Return (reasoning, visible answer) from text that may contain a think block."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip(), strip_think(text)
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
        return reasoning.strip(), rest.strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip(), ""
    return "", text.strip()


def final_segment(text: str) -> str:
    """The part of a free-form reply that most likely is the answer.

    Rule: text after the last "Answer:" style marker if there is one, otherwise
    the last non-empty line.  Models put the conclusion last.
    """
    text = strip_think(text)
    marks = list(_ANSWER_MARK.finditer(text))
    if marks:
        tail = text[marks[-1].end() :].strip()
        return tail.splitlines()[0].strip() if tail else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _unwrap(text: str) -> str:
    text = text.strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.strip()


def norm_integer(text: str) -> str | None:
    found = _INT.findall(final_segment(text))
    return str(int(found[-1])) if found else None


def norm_int_list(text: str) -> str | None:
    """A Python-style list of integers, printed exactly like `str([1, 2, 3])`."""
    segment = strip_think(text)
    blocks = re.findall(r"\[[^\[\]]*\]", segment)
    if not blocks:
        return None
    try:
        values = ast.literal_eval(blocks[-1])
    except (ValueError, SyntaxError):
        return None
    if not isinstance(values, list) or not all(type(v) is int for v in values):
        return None
    return str(values)


def norm_text(text: str) -> str | None:
    """Keep the answer text as written (case and punctuation matter)."""
    out = _unwrap(final_segment(text))
    return out or None


def norm_trivia(text: str) -> str | None:
    """Trivia answers: lower case, no quiz-show lead-in, no trailing full stop."""
    out = _unwrap(final_segment(text)).lower()
    out = _QUESTION_LEAD.sub("", out)
    out = re.sub(r"\s+", " ", out).strip().rstrip(".?!").strip()
    return out or None


def make_label_normalizer(labels: tuple[str, ...]) -> Normalizer:
    """Map a reply onto one of the allowed labels, ignoring case and stray words.

    Longer labels are tried first so that "Not sad" is not mistaken for "Sad".
    """
    by_length = sorted(labels, key=len, reverse=True)
    exact = {label.lower(): label for label in labels}

    def normalize(text: str) -> str | None:
        segment = _unwrap(final_segment(text)).rstrip(".").strip()
        hit = exact.get(segment.lower())
        if hit is not None:
            return hit
        lowered = segment.lower()
        for label in by_length:
            if re.search(rf"(?<![a-z0-9]){re.escape(label.lower())}(?![a-z0-9])", lowered):
                return label
        return None

    return normalize


def extract_code(text: str) -> str | None:
    """Pull Python source out of a reply: the longest fenced block, else the raw text."""
    text = strip_think(text)
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip() or None
    if "```" in text:  # an opening fence that was never closed (output was cut off)
        text = text.split("```", 1)[1]
        text = re.sub(r"^(?:python|py)?[ \t]*\n", "", text, flags=re.IGNORECASE)
    stripped = text.strip()
    return stripped if ("def " in stripped or "import " in stripped) else None
