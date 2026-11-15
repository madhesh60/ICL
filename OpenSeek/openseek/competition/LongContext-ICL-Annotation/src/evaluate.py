# evaluate.py
"""
Local evaluation script for FlagOS Long-Context Data Annotation Competition.

This script tests accuracy LOCALLY before wasting daily submissions on FlagOS.
It supports:
1. Accuracy calculation for datasets 1-7
2. Execution success rate for dataset 8 (code generation)
3. Detailed metrics and error analysis

Usage:
    python evaluate.py --task_id 1
    python evaluate.py --all
"""

import json
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter


# ============================================================================
# Label Extraction Utilities
# ============================================================================

def extract_label_from_prediction(prediction: str) -> str:
    """
    Extract label from prediction string.

    Handles formats:
    - Plain text: "positive"
    - Label tags: "<label>positive</label>"
    - With colon: "Label: positive"
    """
    import re

    # Try XML tags
    match = re.search(r'<label>\s*(.*?)\s*</label>', prediction, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try "Label:" format
    if 'Label:' in prediction:
        parts = prediction.split('Label:')
        return parts[-1].strip().split('\n')[0].strip()

    # Plain text - take first non-empty line
    lines = [l.strip() for l in prediction.split('\n') if l.strip()]
    if lines:
        return lines[0].strip()

    return prediction.strip()


def normalize_for_comparison(label: str) -> str:
    """
    Normalize label for accurate comparison.

    Handles:
    - Case insensitivity
    - Extra whitespace
    - Common variations
    """
    normalized = label.lower().strip()

    # Remove common prefixes/suffixes
    for prefix in ['the label is ', 'label: ', 'answer: ']:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    return normalized.strip()


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calculate_accuracy(
    predictions_file: str,
    ground_truth_file: Optional[str] = None,
    task_data_file: Optional[str] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate accuracy for a dataset.

    Args:
        predictions_file: Path to predictions JSONL file
        ground_truth_file: Optional path to ground truth file
        task_data_file: Optional path to task JSON (for computing from examples)

    Returns:
        Tuple of (accuracy, metrics_dict)
    """
    # Load predictions
    predictions = {}
    with open(predictions_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            pred_id = item.get('id', len(predictions))
            pred_label = extract_label_from_prediction(item.get('prediction', ''))
            predictions[pred_id] = normalize_for_comparison(pred_label)

    # Load ground truth or compute from task data
    ground_truth = {}

    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                gt_id = item.get('id')
                gt_label = item.get('label', '')
                if isinstance(item.get('output'), list):
                    gt_label = item['output'][0]
                ground_truth[gt_id] = normalize_for_comparison(gt_label)

    elif task_data_file and os.path.exists(task_data_file):
        with open(task_data_file, 'r') as f:
            task_data = json.load(f)

        # Use test samples' expected outputs as pseudo-ground truth
        # (This is only for local testing, not real accuracy)
        for sample in task_data.get('test_samples', []):
            sample_id = sample.get('id')
            if 'expected' in sample:
                ground_truth[sample_id] = normalize_for_comparison(sample['expected'])
            elif 'output' in sample:
                gt_label = sample['output'][0] if isinstance(sample['output'], list) else sample['output']
                ground_truth[sample_id] = normalize_for_comparison(gt_label)

    # Calculate metrics
    metrics = {
        'total_predictions': len(predictions),
        'correct': 0,
        'incorrect': 0,
        'missing': 0,
        'error_types': Counter()
    }

    if ground_truth:
        for gt_id, true_label in ground_truth.items():
            if gt_id in predictions:
                if predictions[gt_id] == true_label:
                    metrics['correct'] += 1
                else:
                    metrics['incorrect'] += 1
                    metrics['error_types'][f"got '{predictions[gt_id]}' expected '{true_label}'"] += 1
            else:
                metrics['missing'] += 1

        total = metrics['correct'] + metrics['incorrect']
        accuracy = metrics['correct'] / total if total > 0 else 0.0
    else:
        # No ground truth - just report prediction statistics
        accuracy = None
        metrics['note'] = 'No ground truth available for accuracy calculation'

    return accuracy, metrics


def calculate_dataset8_score(predictions_file: str) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate score for Dataset 8 (code generation).

    Dataset 8 scoring:
    - S_Call: Does the code execute without crashing?
    - S_Execution: Does it produce correct output?

    Args:
        predictions_file: Path to predictions JSONL file

    Returns:
        Tuple of (score, metrics_dict)
    """
    import subprocess
    import sys

    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))

    metrics = {
        'total': len(predictions),
        'executed': 0,
        'crashed': 0,
        'no_output': 0,
        'correct': 0
    }

    for pred in predictions:
        code = pred.get('prediction', '')
        exec_output = pred.get('execution_output', '')

        if not code:
            metrics['crashed'] += 1
            continue

        try:
            result = subprocess.run(
                [sys.executable, '-c', code],
                timeout=10,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                metrics['executed'] += 1
                if exec_output or result.stdout.strip():
                    metrics['correct'] += 1
            else:
                metrics['crashed'] += 1

        except Exception:
            metrics['crashed'] += 1

    # Calculate score (weighted average)
    s_call = metrics['executed'] / metrics['total'] if metrics['total'] > 0 else 0
    s_execution = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0

    # Final score (can be customized based on competition formula)
    score = 0.4 * s_call + 0.6 * s_execution

    return score, metrics


def evaluate_task(
    task_id: int,
    predictions_dir: str,
    data_dir: str = './data'
) -> Dict[str, Any]:
    """
    Evaluate predictions for a single task.

    Args:
        task_id: Dataset ID (1-8)
        predictions_dir: Directory with prediction files
        data_dir: Directory with task data files

    Returns:
        Evaluation results dictionary
    """
    # Find prediction file
    pred_files = [
        f'openseek-{task_id}_predictions.jsonl',
        f'dataset_{task_id}_predictions.jsonl',
        f'openseek-{task_id}-v*.jsonl'
    ]

    predictions_file = None
    for pattern in pred_files:
        if '*' in pattern:
            import glob
            matches = glob.glob(os.path.join(predictions_dir, pattern))
            if matches:
                predictions_file = max(matches)  # Latest version
        else:
            candidate = os.path.join(predictions_dir, pattern)
            if os.path.exists(candidate):
                predictions_file = candidate
                break

    if not predictions_file:
        return {'error': f'No predictions found for task {task_id}'}

    # Find data file
    data_files = {
        1: 'openseek-1_closest_integers.json',
        2: 'openseek-2_count_nouns_verbs.json',
        3: 'openseek-3_collatz_conjecture.json',
        4: 'openseek-4_conala_concat_strings.json',
        5: 'openseek-5_semeval_2018_task1_tweet_sadness_detection.json',
        6: 'openseek-6_mnli_same_genre_classification.json',
        7: 'openseek-7_jeopardy_answer_generation_all.json',
        8: 'openseek-8_kernel_generation.json'
    }

    data_file = os.path.join(data_dir, data_files.get(task_id, f'openseek-{task_id}_*.json'))

    # Evaluate based on dataset type
    if task_id == 8:
        score, metrics = calculate_dataset8_score(predictions_file)
    else:
        accuracy, metrics = calculate_accuracy(
            predictions_file=predictions_file,
            task_data_file=data_file if os.path.exists(data_file) else None
        )
        score = accuracy

    return {
        'task_id': task_id,
        'predictions_file': predictions_file,
        'score': score,
        'metrics': metrics
    }


def print_evaluation(results: Dict[str, Any]):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results - Task {results['task_id']}")
    print(f"{'='*60}")

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    score = results.get('score')
    metrics = results.get('metrics', {})

    if score is not None:
        print(f"Score: {score:.4f} ({score*100:.2f}%)")
    else:
        print("Score: N/A (no ground truth)")

    print(f"\nMetrics:")
    for key, value in metrics.items():
        if key != 'error_types' and key != 'note':
            print(f"  {key}: {value}")

    if metrics.get('error_types'):
        print(f"\nTop errors:")
        for error, count in metrics['error_types'].most_common(5):
            print(f"  ({count}x) {error[:100]}")


# ============================================================================
# Main Execution
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate FlagOS predictions")
    parser.add_argument(
        '--task_id', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help='Task ID to evaluate (1-8)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Evaluate all 8 tasks'
    )
    parser.add_argument(
        '--predictions_dir', type=str,
        default='../outputs',
        help='Directory with prediction files'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='./data',
        help='Directory with task data files'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    if args.all:
        task_ids = list(range(1, 9))
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        print("Please specify --task_id or --all")
        return

    all_results = {}
    for task_id in task_ids:
        results = evaluate_task(
            task_id=task_id,
            predictions_dir=args.predictions_dir,
            data_dir=args.data_dir
        )
        all_results[task_id] = results
        print_evaluation(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_score = 0
    count = 0
    for task_id, results in all_results.items():
        if 'score' in results and results['score'] is not None:
            total_score += results['score']
            count += 1
            print(f"Task {task_id}: {results['score']:.4f}")

    if count > 0:
        avg_score = total_score / count
        print(f"\nAverage Score: {avg_score:.4f} ({avg_score*100:.2f}%)")


if __name__ == "__main__":
    main()
