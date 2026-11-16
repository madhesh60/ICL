"""Stand-in models for tests and dry runs.  None of them is a language model.

    ScriptedBackend    - replies with a prepared list of texts, in order
    FunctionBackend    - replies with whatever a Python function returns
    RouterBackend      - sends each request to the first backend whose rule matches
    DemoReaderBackend  - "reads" the prompt: finds the examples in it, compares them
                         with the input, and answers like the nearest ones

DemoReaderBackend exists so the full pipeline can run end to end on a laptop and
so that prompt-building choices have a visible effect in tests (it prefers
examples that are similar and that sit late in the prompt, as real small models
do).  Numbers produced with it say "the plumbing works".  They say nothing
about how accurate Qwen3-4B is.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence

from cairn.decoding.constraints import Choice
from cairn.llm.base import GenRequest, GenResponse, LLMBackend, Usage
from cairn.retrieval.features import word_tokens

_DEMO = re.compile(r"Input: (.*?)\nOutput:[ \n](.*?)\n\n(?=Input: |# |\Z)", re.DOTALL)


class ScriptedBackend:
    """Returns prepared replies one after another and records what it was asked."""

    model = "scripted"

    def __init__(self, replies: Sequence[str | list[str]]) -> None:
        self._replies = list(replies)
        self.requests: list[GenRequest] = []
        self._lock = threading.Lock()

    def generate(self, request: GenRequest) -> GenResponse:
        with self._lock:
            self.requests.append(request)
            if not self._replies:
                raise AssertionError("ScriptedBackend ran out of prepared replies")
            reply = self._replies.pop(0)
        texts = reply if isinstance(reply, list) else [reply] * request.n
        return GenResponse(texts=list(texts), reasoning=[""] * len(texts))


class FunctionBackend:
    """Wraps a plain function `request -> text | list of texts | GenResponse`."""

    def __init__(
        self, fn: Callable[[GenRequest], str | list[str] | GenResponse], model: str = "function"
    ) -> None:
        self._fn = fn
        self.model = model
        self.calls = 0
        self._lock = threading.Lock()

    def generate(self, request: GenRequest) -> GenResponse:
        with self._lock:
            self.calls += 1
        out = self._fn(request)
        if isinstance(out, GenResponse):
            return out
        texts = out if isinstance(out, list) else [out] * request.n
        return GenResponse(texts=list(texts), reasoning=[""] * len(texts))


class RouterBackend:
    """Tries rules in order; the first rule that returns True handles the request."""

    def __init__(
        self,
        routes: Sequence[tuple[Callable[[GenRequest], bool], LLMBackend]],
        default: LLMBackend,
    ) -> None:
        self._routes = list(routes)
        self._default = default
        self.model = default.model

    def generate(self, request: GenRequest) -> GenResponse:
        for rule, backend in self._routes:
            if rule(request):
                return backend.generate(request)
        return self._default.generate(request)


def _unit_noise(*parts: object) -> float:
    """A repeatable pseudo-random number in [0, 1) made from the arguments."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class DemoReaderBackend:
    """A nearest-example reader that behaves a little like a small ICL model.

    recency   - how much more a late example counts than an early one
    sharpness - how strongly the vote turns into confidence
    noise     - chance of a random slip, to imitate an imperfect model
    """

    model = "demo-reader"

    def __init__(self, recency: float = 2.0, sharpness: float = 4.0, noise: float = 0.0, seed: int = 0):
        self.recency = recency
        self.sharpness = sharpness
        self.noise = noise
        self.seed = seed
        self.calls = 0
        self._parsed: dict[str, list[tuple[Counter[str], str]]] = {}
        self._lock = threading.Lock()

    # -- reading the prompt ---------------------------------------------------

    def _block_demos(self, content: str) -> list[tuple[Counter[str], str]]:
        key = hashlib.sha256(content.encode()).hexdigest()
        with self._lock:
            hit = self._parsed.get(key)
        if hit is None:
            hit = [(Counter(word_tokens(x)), _clean_output(y)) for x, y in _DEMO.findall(content)]
            with self._lock:
                if len(self._parsed) > 64:
                    self._parsed.clear()
                self._parsed[key] = hit
        return hit

    def _read(self, request: GenRequest) -> tuple[list[tuple[Counter[str], str]], str]:
        demos: list[tuple[Counter[str], str]] = []
        messages = request.messages
        for index, (role, content) in enumerate(messages):
            if role != "user":
                continue
            head, _, last_input = content.rpartition("Input: ")
            demos.extend(self._block_demos(head))
            is_last = index == len(messages) - 1
            if not is_last and index + 1 < len(messages) and messages[index + 1][0] == "assistant":
                demos.append((Counter(word_tokens(last_input)), _clean_output(messages[index + 1][1])))
        query = messages[-1][1].rpartition("Input: ")[2]
        query = re.sub(r"\nOutput:\s*$", "", query)
        return demos, query

    # -- answering --------------------------------------------------------------

    def generate(self, request: GenRequest) -> GenResponse:
        with self._lock:
            self.calls += 1
        demos, query = self._read(request)
        q = Counter(word_tokens(query))
        n = len(demos)
        df: Counter[str] = Counter()
        for bag, _ in demos:
            df.update(bag.keys())
        votes: dict[str, float] = {}
        best_output, best_score = "", -1.0
        for position, (bag, output) in enumerate(demos):
            overlap = sum(math.log(1.0 + n / df[w]) for w in q if w in bag)
            late = 1.0 + self.recency * math.exp(-(n - 1 - position) / max(n / 20.0, 1.0))
            score = overlap * late
            votes[output] = votes.get(output, 0.0) + score
            if score > best_score:
                best_output, best_score = output, score

        options = list(request.constraint.options) if isinstance(request.constraint, Choice) else []
        if not options and 0 < len(votes) <= 12:
            options = sorted(votes)
        prompt_tokens = sum(len(c) for _, c in request.messages) // 4
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=4 * request.n)

        if not options:
            texts = [self._wrap(best_output, request) for _ in range(request.n)]
            return GenResponse(texts=texts, reasoning=[""] * request.n, usage=usage)

        total = sum(votes.get(o, 0.0) for o in options) or 1.0
        logits = [self.sharpness * votes.get(o, 0.0) / total for o in options]
        peak = max(logits)
        norm = peak + math.log(sum(math.exp(v - peak) for v in logits))
        logp = {o: v - norm for o, v in zip(options, logits, strict=True)}
        texts, first = [], []
        for sample in range(request.n):
            pick = max(options, key=lambda o: logp[o])
            slip = _unit_noise(self.seed, query, sample, request.temperature, request.seed)
            if self.noise > 0 and slip < self.noise:
                pick = options[int(slip / self.noise * len(options)) % len(options)]
            elif request.temperature > 0:
                # Sample from the distribution with a repeatable random number.
                u, acc = _unit_noise(self.seed, "t", query, sample, request.seed), 0.0
                for o in options:
                    acc += math.exp(logp[o])
                    if u <= acc:
                        pick = o
                        break
            texts.append(self._wrap(pick, request))
            first.append({o.split()[0]: lp for o, lp in logp.items()} if request.top_logprobs else {})
        return GenResponse(texts=texts, reasoning=[""] * request.n, first_token_logprobs=first, usage=usage)

    @staticmethod
    def _wrap(answer: str, request: GenRequest) -> str:
        if request.thinking:
            return f"<think>\nThe closest examples point to this answer.\n</think>\n\n{answer}"
        return answer


def _clean_output(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:python)?\n(.*?)\n?```", text, re.DOTALL)
    return fenced.group(1) if fenced else text
