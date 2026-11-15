# annotate.py
"""
Main annotation script for FlagOS Long-Context Data Annotation Competition.

This script implements the complete annotation pipeline:
1. Load datasets (8 tasks)
2. Build semantic retrieval index from demo examples
3. For each test sample:
   - Retrieve most similar examples using FAISS
   - Build prompt with top/bottom instructions (lost-in-the-middle fix)
   - Run Qwen3-4B via FlagScale API
   - Apply multi-turn consistency with sliding window
4. Save predictions in JSONL format

Usage:
    python annotate.py --task_id 1  # Run specific task
    python annotate.py --all         # Run all 8 tasks
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from retrieval import SemanticRetriever, build_index, get_diverse_demos
from prompt import build_annotation_prompt, extract_label
from dataset8 import annotate_with_correction, extract_code_from_response


# ============================================================================
# STEP 1: LLM Connection (FlagScale API)
# ============================================================================

def get_llm_client(base_url: str = "http://localhost:8000/v1"):
    """
    Get OpenAI-compatible client for FlagScale.

    FlagScale serves Qwen3-4B as a local API server.
    We talk to it like an OpenAI API - send text, get text back.

    Args:
        base_url: FlagScale server URL

    Returns:
        OpenAI client instance
    """
    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",  # FlagScale doesn't need a real API key
        base_url=base_url
    )
    return client


def run_model(
    client,
    prompt: str,
    model_name: str = "Qwen/Qwen3-4B",
    max_tokens: int = 100,
    temperature: float = 0.0
) -> str:
    """
    Send prompt to Qwen3-4B via FlagScale and get response.

    Args:
        client: OpenAI client
        prompt: Input prompt
        model_name: Model identifier
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = deterministic)

    Returns:
        Model response text
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return ""


# ============================================================================
# STEP 2: Dataset Loading
# ============================================================================

def load_task_json(json_path: str) -> Dict[str, Any]:
    """
    Load task definition from JSON file.

    Expected JSON structure:
    {
        "task_id": "openseek-1",
        "task_name": "task_name",
        "Definition": ["Task description"],
        "examples": [{"id": "...", "input": "...", "output": [...]}, ...],
        "test_samples": [{"id": "...", "input": "..."}, ...]
    }

    Args:
        json_path: Path to task JSON file

    Returns:
        Task dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def prepare_demo_examples(task_data: Dict[str, Any], max_examples: int = 100) -> List[Dict]:
    """
    Prepare demo examples from task data.

    Args:
        task_data: Loaded task dictionary
        max_examples: Maximum number of examples to use

    Returns:
        List of example dicts with 'input' and 'label' keys
    """
    examples = task_data.get('examples', [])[:max_examples]
    prepared = []

    for ex in examples:
        # Normalize 'output' to 'label'
        output = ex.get('output', [''])[0] if isinstance(ex.get('output'), list) else ex.get('output', '')
        prepared.append({
            'id': ex.get('id', ''),
            'input': ex.get('input', ''),
            'label': output
        })

    return prepared


def prepare_test_samples(task_data: Dict[str, Any]) -> List[Dict]:
    """
    Prepare test samples from task data.

    Args:
        task_data: Loaded task dictionary

    Returns:
        List of test sample dicts
    """
    samples = task_data.get('test_samples', [])
    prepared = []

    for sample in samples:
        prepared.append({
            'id': sample.get('id', len(prepared)),
            'input': sample.get('input', '')
        })

    return prepared


# ============================================================================
# STEP 3: Annotation Pipeline
# ============================================================================

def annotate_single_sample(
    retriever: SemanticRetriever,
    test_input: str,
    task_description: str,
    annotation_history: List[tuple],
    client,
    dataset_id: int,
    k_similar: int = 4,
    k_diverse: int = 4
) -> Dict[str, Any]:
    """
    Annotate a single test sample.

    Args:
        retriever: Semantic retriever with built index
        test_input: Input text to annotate
        task_description: Task definition
        annotation_history: Sliding window of recent annotations
        client: LLM client
        dataset_id: Which dataset (8 for code generation)
        k_similar: Number of similar examples
        k_diverse: Number of diverse examples

    Returns:
        Prediction result dict
    """
    sample_id = len(annotation_history)

    if dataset_id == 8:
        # Special handling for Dataset 8 - code generation
        demos = get_diverse_demos(
            test_input,
            retriever.demo_set,
            retriever.index,
            k_similar=3,
            k_diverse=3
        )

        generated_code, output = annotate_with_correction(
            None,  # prompt_fn not needed, uses internal
            lambda p: run_model(client, p, max_tokens=2000),
            test_input,
            demos,
            task_description,
            max_retries=3
        )

        return {
            "id": sample_id,
            "input": test_input[:200],  # Truncate for storage
            "prediction": generated_code,
            "execution_output": output
        }

    else:
        # Standard annotation for datasets 1-7
        demos = get_diverse_demos(
            test_input,
            retriever.demo_set,
            retriever.index,
            k_similar=k_similar,
            k_diverse=k_diverse
        )

        # Get recent history for multi-turn consistency
        recent = annotation_history[-3:] if annotation_history else None

        # Build prompt
        prompt = build_annotation_prompt(
            task_description=task_description,
            demos=demos,
            test_input=test_input,
            recent_history=recent
        )

        # Get label from Qwen3-4B
        label = run_model(client, prompt, max_tokens=50)
        label = extract_label(label)

        # Update history
        annotation_history.append((test_input, label))

        return {
            "id": sample_id,
            "input": test_input[:500],  # Truncate for storage
            "prediction": label
        }


def annotate_dataset(
    task_id: int,
    data_path: str,
    output_path: str,
    client,
    max_examples: int = 100,
    k_similar: int = 4,
    k_diverse: int = 4
) -> List[Dict]:
    """
    Annotate all samples in a single dataset.

    Args:
        task_id: Dataset ID (1-8)
        data_path: Path to task JSON file
        output_path: Directory for output files
        client: LLM client
        max_examples: Max demo examples to index
        k_similar: Similar examples per retrieval
        k_diverse: Diverse examples per retrieval

    Returns:
        List of prediction results
    """
    print(f"\n{'='*60}")
    print(f"Processing Dataset {task_id}")
    print(f"{'='*60}")

    # Load task data
    task_data = load_task_json(data_path)
    task_name = task_data.get('task_name', f'dataset_{task_id}')
    task_description = task_data.get('Definition', [''])[0]

    print(f"Task: {task_name}")
    print(f"Description: {task_description[:100]}...")

    # Prepare examples and test samples
    demo_set = prepare_demo_examples(task_data, max_examples)
    test_samples = prepare_test_samples(task_data)

    print(f"Demo examples: {len(demo_set)}")
    print(f"Test samples: {len(test_samples)}")

    # Build retrieval index
    retriever = SemanticRetriever()
    retriever.build_index(demo_set)

    # Multi-turn consistency sliding window
    annotation_history = []

    # Process all test samples
    results = []
    for sample in tqdm(test_samples, desc=f"Dataset {task_id}"):
        test_input = sample['input']
        sample_id = sample['id']

        result = annotate_single_sample(
            retriever=retriever,
            test_input=test_input,
            task_description=task_description,
            annotation_history=annotation_history,
            client=client,
            dataset_id=task_id,
            k_similar=k_similar,
            k_diverse=k_diverse
        )
        result['id'] = sample_id  # Use original sample ID
        results.append(result)

    # Save results
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"openseek-{task_id}_predictions.jsonl")

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Saved {len(results)} predictions to {output_file}")
    return results


# ============================================================================
# STEP 4: Main Execution
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FlagOS Long-Context Data Annotation with Qwen3-4B"
    )
    parser.add_argument(
        '--task_id', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help='Task ID to evaluate (1-8)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all 8 tasks'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='./data',
        help='Directory containing task JSON files'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='../outputs',
        help='Directory for output predictions'
    )
    parser.add_argument(
        '--api_url', type=str,
        default='http://localhost:8000/v1',
        help='FlagScale API URL'
    )
    parser.add_argument(
        '--model', type=str,
        default='Qwen/Qwen3-4B',
        help='Model identifier'
    )
    parser.add_argument(
        '--max_examples', type=int, default=100,
        help='Maximum demo examples to use'
    )
    parser.add_argument(
        '--k_similar', type=int, default=4,
        help='Number of similar examples per retrieval'
    )
    parser.add_argument(
        '--k_diverse', type=int, default=4,
        help='Number of diverse examples per retrieval'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Get LLM client
    print(f"Connecting to FlagScale at {args.api_url}...")
    client = get_llm_client(args.api_url)
    print("Connected successfully.")

    # Determine which tasks to run
    if args.all:
        task_ids = list(range(1, 9))
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        print("Please specify --task_id or --all")
        return

    # Process each task
    all_results = {}
    for task_id in task_ids:
        data_file = os.path.join(args.data_dir, f'openseek-{task_id}_closest_integers.json')
        if task_id == 8:
            data_file = os.path.join(args.data_dir, f'openseek-8_kernel_generation.json')
        elif task_id == 2:
            data_file = os.path.join(args.data_dir, f'openseek-2_count_nouns_verbs.json')
        elif task_id == 3:
            data_file = os.path.join(args.data_dir, f'openseek-3_collatz_conjecture.json')
        elif task_id == 4:
            data_file = os.path.join(args.data_dir, f'openseek-4_conala_concat_strings.json')
        elif task_id == 5:
            data_file = os.path.join(args.data_dir, f'openseek-5_semeval_2018_task1_tweet_sadness_detection.json')
        elif task_id == 6:
            data_file = os.path.join(args.data_dir, f'openseek-6_mnli_same_genre_classification.json')
        elif task_id == 7:
            data_file = os.path.join(args.data_dir, f'openseek-7_jeopardy_answer_generation_all.json')

        if not os.path.exists(data_file):
            print(f"Task file not found: {data_file}")
            continue

        results = annotate_dataset(
            task_id=task_id,
            data_path=data_file,
            output_path=args.output_dir,
            client=client,
            max_examples=args.max_examples,
            k_similar=args.k_similar,
            k_diverse=args.k_diverse
        )
        all_results[task_id] = results

    print(f"\n{'='*60}")
    print("ALL DATASETS ANNOTATED!")
    print(f"{'='*60}")
    print(f"Results saved in: {args.output_dir}/")
    print("\nNext steps:")
    print("1. Run local evaluation: python evaluate.py")
    print("2. Package for submission: python submit.py")
    print("3. Upload to FlagOS platform")


if __name__ == "__main__":
    main()
