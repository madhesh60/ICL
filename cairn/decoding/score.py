"""Reading a label straight from the model's probabilities.

Instead of letting the model write "Sad" and trusting the text, we ask: "for the
very first token of the answer, how likely is each option?"  One token is
generated, nothing has to be parsed, there is never an empty answer, and we get
a confidence number for free.

Limitation: every option must start with a different piece of text ("Sad" vs
"Not", "Y" vs "N", "0".."6").  `distinct_starts` checks this up front.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cairn.llm.base import GenRequest, LLMBackend


def distinct_starts(options: Sequence[str]) -> bool:
    """True if no option's first word is a prefix of another option's first word."""
    heads = [o.split()[0] if o.split() else "" for o in options]
    for i, a in enumerate(heads):
        for j, b in enumerate(heads):
            if i != j and (not a or b.startswith(a)):
                return False
    return True


def options_from_logprobs(first_token: dict[str, float], options: Sequence[str]) -> np.ndarray:
    """Turn {token: logprob} for the first answer token into log-probabilities per option.

    The server may split a label into pieces ("Not" + " sad") and may add a space
    in front.  A token is credited to an option when the option starts with it.
    A token that fits several options (or none) is ignored.  An option that is
    not among the reported tokens gets a floor just below the smallest reported
    value: "unlikely", but never "impossible".
    """
    mass = np.full(len(options), -np.inf)
    for token, logprob in first_token.items():
        piece = token.strip()
        if not piece:
            continue
        fits = [i for i, option in enumerate(options) if option.startswith(piece)]
        if len(fits) == 1:
            mass[fits[0]] = np.logaddexp(mass[fits[0]], logprob)
    reported = [lp for lp in first_token.values() if math.isfinite(lp)]
    floor = (min(reported) if reported else -20.0) - 1.0
    mass[~np.isfinite(mass)] = floor
    return mass - _logsumexp(mass)


def _logsumexp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + float(np.log(np.sum(np.exp(values - peak))))


def score_options(
    backend: LLMBackend,
    messages: list[dict[str, str]],
    options: Sequence[str],
    *,
    top_logprobs: int = 20,
    tag: str = "score",
) -> tuple[np.ndarray, str]:
    """Ask for one token and return (log-probabilities per option, raw text)."""
    request = GenRequest.from_messages(
        messages,
        max_tokens=1,
        temperature=0.0,
        thinking=False,
        top_logprobs=top_logprobs,
        tag=tag,
    )
    response = backend.generate(request)
    first = response.first_token_logprobs[0] if response.first_token_logprobs else {}
    if not first:
        raise ValueError("the server returned no log-probabilities; enable logprobs or use generation")
    return options_from_logprobs(first, options), response.texts[0]
