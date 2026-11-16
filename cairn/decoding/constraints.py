"""What shape an answer is allowed to have.

A constraint is used twice:
  * sent to the server, so the model *cannot* write anything else
    ("structured outputs"), and
  * checked again on our side, because servers differ and "trust but verify"
    costs one line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Choice:
    """The answer must be exactly one of these strings."""

    options: tuple[str, ...]


@dataclass(frozen=True)
class Regex:
    """The whole answer must match this pattern."""

    pattern: str


Constraint = Choice | Regex


def satisfies(constraint: Constraint | None, text: str) -> bool:
    if constraint is None:
        return bool(text.strip())
    if isinstance(constraint, Choice):
        return text in constraint.options
    return re.fullmatch(constraint.pattern, text, flags=re.DOTALL) is not None


def to_request_fields(constraint: Constraint | None, api_style: str) -> dict[str, Any]:
    """Translate a constraint into the JSON fields a vLLM server understands.

    vLLM changed the field names over time, so the style is configurable:
      "structured_outputs"  - vLLM 0.12 and newer: {"structured_outputs": {"choice": [...]}}
      "guided"              - older vLLM:          {"guided_choice": [...]}
      "none"                - server cannot constrain; we rely on the prompt and the parser
    """
    if constraint is None or api_style == "none":
        return {}
    kind, value = (
        ("choice", list(constraint.options))
        if isinstance(constraint, Choice)
        else ("regex", constraint.pattern)
    )
    if api_style == "structured_outputs":
        return {"structured_outputs": {kind: value}}
    if api_style == "guided":
        return {f"guided_{kind}": value}
    raise ValueError(f"unknown structured-output api style: {api_style!r}")
