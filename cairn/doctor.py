"""`cairn doctor`: ask the server what it can do before a long run depends on it.

Cairn uses five server features.  Each one can be missing or named differently
depending on the server version.  Finding that out at minute 0 with six tiny
requests is much better than finding it out at hour 3.

Every feature has a fallback, so a failed check is a warning with advice, not a
dead end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from cairn.decoding.constraints import Choice
from cairn.llm.base import BadRequestError, GenRequest, LLMError
from cairn.llm.openai_compat import OpenAICompatBackend


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    advice: str = ""


def _ask(backend: OpenAICompatBackend, content: str, **kwargs: object) -> object:
    return backend.generate(GenRequest(messages=(("user", content),), tag="doctor", **kwargs))  # type: ignore[arg-type]


def run_checks(backend: OpenAICompatBackend) -> list[Check]:
    checks: list[Check] = []

    try:
        models = backend.list_models()
        known = backend.model in models
        checks.append(
            Check(
                "server reachable",
                True,
                f"models: {', '.join(models) or 'none listed'}",
                "" if known else f"model '{backend.model}' is not listed; set backend.model to one of the above",
            )
        )
    except Exception as err:
        checks.append(Check("server reachable", False, str(err)[:200], "start the server or fix backend.base_url"))
        return checks

    try:
        reply = _ask(backend, "Reply with the single word OK.", max_tokens=8)
        text = reply.texts[0]  # type: ignore[attr-defined]
        clean = bool(text) and "<think>" not in text
        checks.append(
            Check(
                "plain answer (thinking off)",
                clean,
                f"got {text[:40]!r}",
                "" if clean else "thinking could not be switched off; check chat_template_kwargs support",
            )
        )
    except LLMError as err:
        checks.append(Check("plain answer (thinking off)", False, str(err)[:200]))

    try:
        reply = _ask(backend, "Reply with the single word OK.", max_tokens=1, top_logprobs=5)
        first = reply.first_token_logprobs[0]  # type: ignore[attr-defined]
        checks.append(
            Check(
                "token probabilities (logprobs)",
                bool(first),
                f"{len(first)} candidates returned",
                "" if first else "label scoring is unavailable; Cairn will generate labels instead",
            )
        )
    except LLMError as err:
        checks.append(Check("token probabilities (logprobs)", False, str(err)[:200], "set strategy.use_scoring: false"))

    working_style = None
    for style in (backend.structured_api, "structured_outputs", "guided"):
        if style == "none":
            continue
        probe = OpenAICompatBackend(
            backend.base_url, backend.model, structured_api=style, retry=backend.retry, timeout_s=backend.timeout_s
        )
        try:
            reply = _ask(probe, "Pick one.", max_tokens=8, constraint=Choice(("alpha", "beta")))
            if reply.texts[0] in {"alpha", "beta"}:  # type: ignore[attr-defined]
                working_style = style
                break
        except (BadRequestError, LLMError):
            continue
    checks.append(
        Check(
            "constrained answers (structured outputs)",
            working_style is not None,
            f"works with structured_api: {working_style}" if working_style else "no style worked",
            ""
            if working_style == backend.structured_api
            else (
                f"set backend.structured_api: {working_style}"
                if working_style
                else "set backend.structured_api: none (Cairn then relies on its parser)"
            ),
        )
    )

    try:
        reply = _ask(backend, "What is 17 + 26? Think, then answer.", max_tokens=512, thinking=True)
        thought = reply.reasoning[0]  # type: ignore[attr-defined]
        checks.append(
            Check(
                "thinking mode",
                bool(thought),
                f"{len(thought)} characters of reasoning, answer {reply.texts[0][:20]!r}",  # type: ignore[attr-defined]
                "" if thought else "no reasoning came back; escalation will behave like plain sampling",
            )
        )
    except LLMError as err:
        checks.append(Check("thinking mode", False, str(err)[:200]))

    try:
        filler = "The quick brown fox jumps over the lazy dog. " * 400
        stamp = f"[probe {time.time():.0f}] "
        first = _ask(backend, stamp + filler + "Reply with the single word A.", max_tokens=2)
        second = _ask(backend, stamp + filler + "Reply with the single word B.", max_tokens=2)
        cached = second.usage.cached_tokens  # type: ignore[attr-defined]
        faster = second.latency_s < 0.6 * first.latency_s  # type: ignore[attr-defined]
        ok = cached > 0 or faster
        checks.append(
            Check(
                "prefix cache",
                ok,
                f"second call: {cached} cached tokens, {first.latency_s:.2f}s -> {second.latency_s:.2f}s",  # type: ignore[attr-defined]
                ""
                if ok
                else "enable prefix caching on the server (vLLM: remove --no-enable-prefix-caching); "
                "runs still work but re-read the whole prompt every time",
            )
        )
    except LLMError as err:
        checks.append(Check("prefix cache", False, str(err)[:200]))
    return checks
