"""How one example, and the fixed instruction text, are written into the prompt.

Two rules shape this file:

1. Every example ends with a blank line.  Qwen's tokenizer never glues tokens
   across a line break, so the token count of N examples joined together equals
   the sum of their separate counts.  That lets us budget by simple addition.
   (A test checks this on real data.)

2. The instruction text contains no made-up sample answers.  The official
   baseline wrote "<label>Good Review</label>" as a format illustration and the
   model then answered "Good Review" on tasks that have nothing to do with reviews.
"""

from __future__ import annotations

from cairn.tasks.base import TaskSpec
from cairn.types import Example

BULK_HEADING = "# Reference examples\nEach example shows an input and its correct output.\n\n"
SESSION_HEADING = "# Labelled earlier in this session\n\n"
TAIL_HEADING = "# Closest examples to the next input\n\n"
TURNS_BRIDGE = "# Now label new inputs, one at a time.\n\n"


def render_input(text: str) -> str:
    return f"Input: {text}"


def render_output(spec: TaskSpec, output: str) -> str:
    if spec.output_style == "code":
        return f"```python\n{output.strip()}\n```"
    return output


def render_demo(spec: TaskSpec, example: Example) -> str:
    """One example as a text block that ends with a blank line."""
    if spec.output_style == "code":
        return f"{render_input(example.input)}\nOutput:\n{render_output(spec, example.output)}\n\n"
    return f"{render_input(example.input)}\nOutput: {example.output}\n\n"


def system_prompt(spec: TaskSpec, definition: str, guidelines: str = "") -> str:
    parts = [
        "You are a careful data annotator. You label each input exactly the way the "
        "reference examples are labelled.",
        f"# Task\n{definition.strip()}",
        f"# Output format\n{spec.answer_rule}",
    ]
    if guidelines.strip():
        parts.append(f"# Guidelines learned from the examples\n{guidelines.strip()}")
    return "\n\n".join(parts)
