# prompt.py
"""
Prompt template builder for long-context data annotation.
Implements optimized prompt design with instructions at top and bottom (lost-in-the-middle fix).
"""

from typing import List, Dict, Any, Optional


def build_annotation_prompt(
    task_description: str,
    demos: List[Dict[str, Any]],
    test_input: str,
    recent_history: Optional[List[tuple]] = None,
    mode: str = "standard"
) -> str:
    """
    Build the full prompt for annotation.

    IMPORTANT STRATEGY FROM THE HACKATHON PLAN:
    - Most important instructions go at TOP and BOTTOM
    - Examples go in the MIDDLE (less critical)
    - This is because the AI forgets the middle (lost-in-the-middle problem)

    Args:
        task_description: Clear description of the annotation task
        demos: List of demo examples with 'input' and 'label'/'output' keys
        test_input: The text to be annotated
        recent_history: Optional list of (input, label) tuples for multi-turn consistency
        mode: Prompt mode - "standard" or "chain_of_thought"

    Returns:
        Complete prompt string ready for LLM
    """
    # TOP: THE MOST IMPORTANT INSTRUCTIONS
    prompt = """[SYSTEM INSTRUCTION]
You are a precise data annotation assistant.
Your ONLY job is to output the correct label for the given input.
Follow the task definition exactly. Do not add explanations.
Think step by step before giving your final label.
Respond ONLY with the label itself. Nothing else.

[TASK DEFINITION]
""" + task_description + "\n\n"

    # MIDDLE: EXAMPLES (less critical, goes in centre)
    prompt += "[ANNOTATION EXAMPLES]\n"

    # Add recent history for multi-turn consistency
    if recent_history:
        prompt += "[RECENT ANNOTATIONS FOR CONSISTENCY]\n"
        for hist_input, hist_label in recent_history:
            prompt += f"Input: {hist_input}\nLabel: {hist_label}\n\n"

    # Add the retrieved demo examples
    for i, demo in enumerate(demos, 1):
        input_text = demo['input']
        # Handle both 'label' and 'output' keys
        output_text = demo.get('label') or (demo.get('output', [''])[0] if isinstance(demo.get('output'), list) else demo.get('output', ''))
        prompt += f"Example {i}:\n{input_text}\nLabel: {output_text}\n\n"

    # BOTTOM: REPEAT INSTRUCTIONS + THE TARGET
    # Repeat key instruction at bottom because AI remembers the end
    prompt += f"""[TARGET SAMPLE - ANNOTATE THIS]
REMINDER: Output ONLY the label. Nothing else. No explanation.
Input: {test_input}
Label:"""

    return prompt


def build_annotation_prompt_xml(
    task_description: str,
    demos: List[Dict[str, Any]],
    test_input: str,
    recent_history: Optional[List[tuple]] = None
) -> str:
    """
    Build prompt with XML-style tags for better structure.

    Uses XML tags to improve readability and parsing:
    - <task> for task definition
    - <examples> for demonstration examples
    - <sample> for the input to annotate

    Args:
        task_description: Task description
        demos: Demo examples
        test_input: Input to annotate
        recent_history: Optional recent annotation history

    Returns:
        XML-structured prompt
    """
    prompt = """<system>
You are a precise data annotation assistant.
Follow the task definition and examples exactly.
Output ONLY the final label, wrapped in <label> tags.
</system>

<task>
""" + task_description + """
</task>

<examples>"""

    if recent_history:
        prompt += "\n<!-- Recent annotations for consistency -->"
        for hist_input, hist_label in recent_history:
            prompt += f"""
<example type="history">
<input>{hist_input}</input>
<label>{hist_label}</label>
</example>"""

    for i, demo in enumerate(demos, 1):
        input_text = demo['input']
        output_text = demo.get('label') or (demo.get('output', [''])[0] if isinstance(demo.get('output'), list) else demo.get('output', ''))
        prompt += f"""
<example id="{i}">
<input>{input_text}</input>
<label>{output_text}</label>
</example>"""

    prompt += """
</examples>

<sample to annotate>
""" + test_input + """
</sample>

<output_format>
Output ONLY the label in <label></label> tags. No other text.
</output_format>

Label:"""

    return prompt


def build_code_generation_prompt(
    task_description: str,
    demos: List[Dict[str, Any]],
    test_input: str
) -> str:
    """
    Special prompt for Dataset 8 - code generation.

    Dataset 8 is evaluated on code execution, not just label accuracy.
    The goal is to generate Python code that RUNS correctly.

    Args:
        task_description: Task description for code generation
        demos: Demo examples with code solutions
        test_input: The problem statement

    Returns:
        Prompt for code generation
    """
    prompt = """[SYSTEM INSTRUCTION]
You are a Python code generation assistant.
Generate ONLY executable Python code. No explanation, no markdown, no comments.
The code must run without errors and print the correct numerical result.
Include input validation. Handle edge cases.

[TASK DEFINITION]
""" + task_description + """

[CODE EXAMPLES]"""

    for i, demo in enumerate(demos, 1):
        input_text = demo['input']
        output_text = demo.get('label') or (demo.get('output', [''])[0] if isinstance(demo.get('output'), list) else demo.get('output', ''))
        prompt += f"""
Example {i}:
Problem: {input_text}
Code:
{output_text}
"""

    prompt += """[TARGET PROBLEM]
REMINDER: Output ONLY valid Python code. It must execute and print the correct answer.
Problem: """ + test_input + """

Code:"""

    return prompt


def build_simple_prompt(
    task_description: str,
    demos: List[Dict[str, Any]],
    test_input: str
) -> str:
    """
    Simple prompt template for quick testing.

    Args:
        task_description: Task description
        demos: Demo examples
        test_input: Input to annotate

    Returns:
        Simple prompt string
    """
    prompt = f"""Task: {task_description}

Examples:
"""
    for demo in demos:
        input_text = demo['input']
        output_text = demo.get('label') or (demo.get('output', [''])[0] if isinstance(demo.get('output'), list) else demo.get('output', ''))
        prompt += f"Input: {input_text}\nOutput: {output_text}\n\n"

    prompt += f"""Now annotate this:
Input: {test_input}
Output:"""

    return prompt


def extract_label(response: str) -> str:
    """
    Extract label from LLM response.

    Handles various formats:
    - Plain text label
    - <label>...</label> tags
    - Label at end of response

    Args:
        response: LLM response text

    Returns:
        Extracted label string
    """
    import re

    # Try XML tags first
    match = re.search(r'<label>\s*(.*?)\s*</label>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find "Label:" at end
    if 'Label:' in response:
        label = response.split('Label:')[-1].strip()
        # Take first line only
        return label.split('\n')[0].strip()

    # Fall back to first line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        return lines[0].strip()

    return response.strip()
