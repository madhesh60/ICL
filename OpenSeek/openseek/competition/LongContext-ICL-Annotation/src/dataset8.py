# dataset8.py
"""
Special handler for Dataset 8 (Triton Kernel Code Generation).

Dataset 8 is special because it's evaluated on code execution, not just label accuracy.
The scoring formula rewards:
1. Code that runs without crashing (S_Call)
2. Code that produces the correct output (S_Execution)

This module implements the self-correction loop: generate code, run it, fix if it crashes.
"""

import subprocess
import sys
import re
from typing import Tuple, Optional, Callable


def run_code_safely(code_string: str, timeout: int = 10) -> Tuple[bool, str, str]:
    """
    Run generated Python code and check execution status.

    This validates:
    1. Does it run without crashing? (S_Call score)
    2. Does it produce output? (S_Execution score)

    Args:
        code_string: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success, output, error)
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', code_string],
            timeout=timeout,
            capture_output=True,
            text=True
        )
        success = result.returncode == 0
        output = result.stdout.strip()
        error = result.stderr.strip()
        return success, output, error

    except subprocess.TimeoutExpired:
        return False, "", "Code took too long to run (timeout)"
    except Exception as e:
        return False, "", str(e)


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.

    Handles:
    - Code in markdown code blocks
    - Raw code without formatting
    - Code with explanation text

    Args:
        response: LLM response containing code

    Returns:
        Extracted Python code string
    """
    # Try to find code in markdown code blocks
    code_patterns = [
        r'```python\n(.*?)\n```',  # ```python ... ```
        r'```\n(.*?)\n```',         # ``` ... ```
        r'`(.*?)`',                  # Single line code
    ]

    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no code block found, assume entire response is code
    return response.strip()


def verify_code_correctness(
    code: str,
    expected_output: Optional[str] = None,
    test_cases: Optional[list] = None
) -> Tuple[bool, str]:
    """
    Verify code produces correct results.

    Args:
        code: Python code to test
        expected_output: Expected output string
        test_cases: Optional list of test case dicts with input/expected

    Returns:
        Tuple of (is_correct, message)
    """
    success, output, error = run_code_safely(code)

    if not success:
        return False, f"Execution failed: {error}"

    if test_cases:
        # Run through test cases
        for i, tc in enumerate(test_cases):
            test_code = f"{code}\n\n# Test case {i+1}\nprint({tc['input']})"
            success, output, error = run_code_safely(test_code)
            if not success or output != tc['expected']:
                return False, f"Test case {i+1} failed"

    elif expected_output:
        if output != expected_output:
            return False, f"Output mismatch: got '{output}', expected '{expected_output}'"

    return True, "Code verified successfully"


def annotate_with_correction(
    prompt_fn: Callable,
    run_model_fn: Callable,
    test_input: str,
    demos: list,
    task_desc: str,
    max_retries: int = 3
) -> Tuple[str, str]:
    """
    Self-correction loop for Dataset 8 code generation.

    Step 1: Generate code
    Step 2: Run it
    Step 3: If it crashes, tell the AI the error and ask it to fix
    Step 4: Repeat up to max_retries times

    This is the BIG WINNING TRICK for Dataset 8 - self-correction significantly
    boosts S_Execution score.

    Args:
        prompt_fn: Function to build code generation prompt
        run_model_fn: Function to call LLM
        test_input: Problem statement
        demos: Demo examples
        task_desc: Task description
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (generated_code, execution_output)
    """
    from prompt import build_code_generation_prompt

    # First attempt
    prompt = build_code_generation_prompt(task_desc, demos, test_input)
    generated_code = run_model_fn(prompt)
    generated_code = extract_code_from_response(generated_code)

    for attempt in range(max_retries):
        success, output, error = run_code_safely(generated_code)

        if success and output:
            # Code ran fine! Return it.
            print(f"  [Dataset8] Code worked on attempt {attempt + 1}")
            return generated_code, output

        # Code failed. Tell the AI what went wrong and ask it to fix.
        print(f"  [Dataset8] Attempt {attempt + 1} failed: {error[:100] if error else 'No output'}")

        if attempt < max_retries - 1:
            # Build fix prompt with error message
            fix_prompt = f"""The following Python code has an error:

```
{generated_code}
```

Error message: {error}

Fix the code. Output ONLY the corrected Python code. No explanation.
The fixed code must run without errors and print the correct result.

Code:"""

            generated_code = run_model_fn(fix_prompt)
            generated_code = extract_code_from_response(generated_code)

    # All attempts failed, return best effort
    print(f"  [Dataset8] All {max_retries} attempts failed, returning best effort")
    return generated_code, ""


def generate_kernel_code(
    run_model_fn: Callable,
    problem_description: str,
    examples: list,
    max_tokens: int = 2000
) -> str:
    """
    Specialized kernel code generator for Dataset 8 Triton tasks.

    Args:
        run_model_fn: LLM call function
        problem_description: Triton kernel problem description
        examples: Example Triton kernels
        max_tokens: Maximum tokens for generation

    Returns:
        Generated Python code
    """
    prompt = f"""You are an expert in writing Triton operators for efficient GPU programming.

Task: {problem_description}

Generate ONLY valid Python code with Triton. The code must:
1. Define all Triton kernels using @triton.jit decorator
2. Include the Python wrapper function to call the kernel
3. Use proper Triton's tl.* operations
4. Handle edge cases and boundary conditions

Code Examples:
{chr(10).join([f'Example {i+1}: {ex["input"][:200]}... -> Code' for i, ex in enumerate(examples[:3])])}

Generate the complete executable code now:
"""

    response = run_model_fn(prompt, max_tokens=max_tokens)
    return extract_code_from_response(response)


def validate_triton_code(code: str) -> Tuple[bool, str]:
    """
    Validate that generated code is valid Triton code.

    Args:
        code: Python code to validate

    Returns:
        Tuple of (is_valid, message)
    """
    checks = [
        ('import torch' in code, 'Missing torch import'),
        ('import triton' in code, 'Missing triton import'),
        ('@triton.jit' in code, 'Missing @triton.jit decorator'),
        ('def ' in code, 'Missing function definition'),
        ('tl.load' in code or 'tl.store' in code, 'Missing Triton memory operations'),
    ]

    for condition, message in checks:
        if not condition:
            return False, message

    return True, "Valid Triton code structure"


def get_dataset8_score(execution_output: str, expected_output: str) -> float:
    """
    Calculate score for Dataset 8 based on execution output matching.

    Args:
        execution_output: What the code produced
        expected_output: What was expected

    Returns:
        Score between 0 and 1
    """
    # Normalize outputs for comparison
    exec_normalized = execution_output.strip().lower()
    expected_normalized = expected_output.strip().lower()

    # Exact match
    if exec_normalized == expected_normalized:
        return 1.0

    # Try numeric comparison
    try:
        exec_num = float(exec_normalized)
        expected_num = float(expected_normalized)
        if abs(exec_num - expected_num) < 1e-6:
            return 1.0
        elif abs(exec_num - expected_num) < 1.0:
            return 0.5
    except ValueError:
        pass

    return 0.0
