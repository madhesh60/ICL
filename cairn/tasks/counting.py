"""Task 2: counting nouns or verbs by listing them first.

Why a separate rung for counting
--------------------------------
A count is the length of a list.  A small model asked for the number directly guesses
it from the sentence's shape (E001: four samples agreed and were right 43% of the time).
Asked to think, it wanders: E002's thinking hit its 768-token limit on 124 of 200
samples, arguing about typos and earlier examples instead of answering.

So the model does the part it is good at, naming the words, and code does the part code
is good at, counting them:

    model   "Nouns: man, horse, beach"      (constrained format, no thinking, ~20 tokens)
    code    len(["man", "horse", "beach"])  ->  "3"

Several sampled lists vote on the count.  The format is enforced by the server
(structured output), so every reply can be read.

The counting rules
------------------
The labels follow a tagger-like convention that is not in the task definition.  It was
measured, not guessed (docs/08_THIRD_RUN.md section 3):

* a least-squares fit of each verb label on the words of its sentence gives "is" a
  weight of +0.02 and "are" -0.02 (2,755 sentences): helper verbs add nothing;
* "living" gets +0.81 on the noun side: each word of a compound noun counts;
* reading labelled sentences confirms it: "A tennis player holding a racket on the
  tennis court" has 5 nouns (tennis, player, racket, tennis, court); "A man is using a
  cell phone to photograph a barn" has 2 verbs (using, photograph).

The rules go into the system prompt, so every rung (listing, thinking) reads them.
"""

from __future__ import annotations

import re

_T2 = re.compile(r"^Sentence: '(.*)'\. Count the number of (nouns|verbs) in this sentence\.$", re.S)
_LIST = re.compile(r"(?:Nouns|Verbs)\s*:\s*(.*)", re.IGNORECASE)
_WORD = r"[A-Za-z][A-Za-z'\-]*"

COUNTING_RULES = """\
- Nouns: count every noun word separately. Each word of a compound noun counts on its own: \
"tennis court" is 2 nouns, "living room" is 2, "stop sign" is 2, "fire hydrant" is 2.
- "front" in "in front of" and "top" in "on top of" are nouns. "next" in "next to" is not.
- Pronouns (it, them, he, she, they, someone) are not nouns.
- A noun that appears twice counts twice.
- Verbs: count main verbs, including -ing and -ed forms used as verbs: "a man riding a \
horse" has 1 verb (riding); "a bus parked by a curb" has 1 (parked).
- Helping verbs in front of another verb are NOT counted: "is", "are", "was", "were", "be", \
"been", "being" in "is sitting", "are being canned", "is painted".
- A word used as a noun or adjective is not a verb: "bowling" in "bowling alley", \
"living" in "living room"."""


def list_instruction(text: str) -> tuple[str, str] | None:
    """The request added after the input, and the regex its reply must match.

    Returns None when the input is not a counting question (the rung is then skipped).
    """
    match = _T2.match(text)
    if not match:
        return None
    kind = match.group(2)
    label = kind.capitalize()
    instruction = (
        f"\n\nDo not answer with a number yet. First list the {kind} of the sentence in the "
        f"order they appear, following the counting rules, in exactly this form:\n"
        f"{label}: word, word, word\n"
        "Write 'none' if there are none. Repeat a word if it appears twice."
    )
    regex = rf"{label}: (none|{_WORD}(, {_WORD}){{0,11}})"
    return instruction, regex


def count_from_list(reply: str) -> str | None:
    """Turn "Nouns: man, horse, beach" into "3".  None if the reply is not a list."""
    match = _LIST.search(reply)
    if not match:
        return None
    body = match.group(1).strip().splitlines()[0].strip().rstrip(".")
    if body.lower() in {"none", ""}:
        return "0"
    items = [item.strip() for item in body.split(",") if item.strip()]
    if not all(re.fullmatch(_WORD + r"(?: " + _WORD + r")*", item) for item in items):
        return None
    # Words, not items: an unconstrained reply may write "tennis court" as one item, and
    # the labels count it as two nouns.
    return str(sum(len(item.split()) for item in items))
