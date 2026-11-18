"""Task 2: listing then counting, and the counting rules checked against the real labels."""

import re

import pytest

from cairn.tasks.counting import COUNTING_RULES, count_from_list, list_instruction

NOUN_Q = "Sentence: 'A man riding a white horse on a sandy beach'. Count the number of nouns in this sentence."
VERB_Q = "Sentence: 'A man is using a cell phone to photograph a barn'. Count the number of verbs in this sentence."


def test_instruction_names_the_kind_and_constrains_the_format():
    instruction, pattern = list_instruction(NOUN_Q)
    assert "list the nouns" in instruction and "Nouns: word, word, word" in instruction
    assert re.fullmatch(pattern, "Nouns: man, horse, beach")
    assert re.fullmatch(pattern, "Nouns: none")
    assert not re.fullmatch(pattern, "3")  # a bare number is not allowed: list first
    _, verb_pattern = list_instruction(VERB_Q)
    assert re.fullmatch(verb_pattern, "Verbs: using, photograph")
    assert list_instruction("not a counting question") is None


@pytest.mark.parametrize(
    ("reply", "count"),
    [
        ("Nouns: man, horse, beach", "3"),
        ("Verbs: none", "0"),
        ("Nouns: tennis, player, racket, tennis, court", "5"),  # repeats count twice
        ("Nouns: tennis court, living room", "4"),  # words, not items (the label convention)
        ("Verbs: using, photograph.", "2"),
        ("I think there are 3", None),  # not a list: unreadable, the ladder takes over
    ],
)
def test_count_from_list(reply, count):
    assert count_from_list(reply) == count


@pytest.mark.data
def test_the_counting_rules_agree_with_the_labels(task_loader):
    """Every worked example quoted in the rules must carry exactly the stated label in the data.

    The rules were derived from the data (a regression of labels on words, then reading
    labelled sentences).  This test keeps them honest: if a quoted sentence's label differs,
    the rule is wrong, not the data.
    """
    labels = {e.input: e.output for e in task_loader(2).examples}

    def label(sentence: str, kind: str) -> str:
        return labels[f"Sentence: '{sentence}'. Count the number of {kind} in this sentence."]

    assert label("A tennis player holding a racket on the tennis court", "nouns") == "5"
    assert label("a living room with some brick walls and a fireplace", "nouns") == "5"
    assert label("A bench sitting in front of a wall and next to a door", "nouns") == "4"
    assert label("A diner with large pepsi signs on the front of it", "nouns") == "4"
    assert label("A man is using a cell phone to photograph a barn", "verbs") == "2"
    assert label("A elephant that is standing on a floor at a bowling alley", "verbs") == "1"
    assert label("Many teddy bears are displayed in front of the trees", "verbs") == "1"
    assert "tennis court" in COUNTING_RULES and '"is sitting"' in COUNTING_RULES
