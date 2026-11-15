# submit.py
"""
Submission packaging script for FlagOS Long-Context Data Annotation Competition.

This script:
1. Collects all prediction files from all 8 datasets
2. Validates file completeness
3. Packages everything into a ZIP file for FlagOS upload

Usage:
    python submit.py                          # Package all predictions
    python submit.py --predictions_dir ./predictions  # Custom directory
"""

import zipfile
import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional


# ============================================================================
# Submission File Validation
# ============================================================================

def validate_prediction_file(filepath: str, min_samples: int = 10) -> Dict[str, any]:
    """
    Validate a prediction file for submission.

    Args:
        filepath: Path to prediction JSONL file
        min_samples: Minimum expected samples

    Returns:
        Validation results dict
    """
    if not os.path.exists(filepath):
        return {
            'valid': False,
            'error': 'File not found'
        }

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < min_samples:
            return {
                'valid': False,
                'error': f'Only {len(lines)} samples (expected at least {min_samples})'
            }

        # Validate JSON format
        for i, line in enumerate(lines[:5]):  # Check first 5
            try:
                item = json.loads(line)
                if 'id' not in item or 'prediction' not in item:
                    return {
                        'valid': False,
                        'error': f'Line {i+1}: Missing required fields'
                    }
            except json.JSONDecodeError as e:
                return {
                    'valid': False,
                    'error': f'Line {i+1}: Invalid JSON - {e}'
                }

        return {
            'valid': True,
            'samples': len(lines),
            'file': os.path.basename(filepath)
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def find_prediction_files(predictions_dir: str) -> Dict[int, str]:
    """
    Find all prediction files for the 8 datasets.

    Args:
        predictions_dir: Directory to search

    Returns:
        Dict mapping task_id to filepath
    """
    files = {}

    # Possible naming patterns
    patterns = [
        'openseek-{id}_predictions.jsonl',
        'openseek-{id}-v*.jsonl',  # Versioned
        'dataset_{id}_predictions.jsonl',
    ]

    for task_id in range(1, 9):
        for pattern in patterns:
            pattern_str = pattern.format(id=task_id)

            if '*' in pattern_str:
                import glob
                matches = glob.glob(os.path.join(predictions_dir, pattern_str))
                if matches:
                    files[task_id] = max(matches)  # Latest version
                    break
            else:
                candidate = os.path.join(predictions_dir, pattern_str)
                if os.path.exists(candidate):
                    files[task_id] = candidate
                    break

    return files


# ============================================================================
# ZIP Packaging
# ============================================================================

def create_submission_zip(
    prediction_files: Dict[int, str],
    output_path: str,
    include_metadata: bool = True
) -> str:
    """
    Create submission ZIP file.

    Args:
        prediction_files: Dict of task_id -> filepath
        output_path: Output ZIP file path
        include_metadata: Whether to include metadata file

    Returns:
        Path to created ZIP file
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        added_files = []

        for task_id, filepath in sorted(prediction_files.items()):
            filename = f'openseek-{task_id}_predictions.jsonl'
            zipf.write(filepath, filename)
            added_files.append(filename)
            print(f"  Added: {filename}")

        # Add metadata
        if include_metadata:
            metadata = {
                'created_at': datetime.now().isoformat(),
                'datasets': list(prediction_files.keys()),
                'files': added_files,
                'total_datasets': len(prediction_files)
            }
            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr('metadata.json', metadata_json)
            print("  Added: metadata.json")

    return output_path


# ============================================================================
# Submission Report
# ============================================================================

def generate_submission_report(
    prediction_files: Dict[int, str],
    output_path: str
) -> str:
    """
    Generate a submission report.

    Args:
        prediction_files: Found prediction files
        output_path: Report file path

    Returns:
        Report content
    """
    report = []
    report.append("=" * 60)
    report.append("FlagOS Submission Report")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("Prediction Files:")
    report.append("-" * 40)

    total_samples = 0
    for task_id in sorted(prediction_files.keys()):
        filepath = prediction_files[task_id]
        validation = validate_prediction_file(filepath)

        samples = validation.get('samples', 0)
        total_samples += samples

        status = "OK" if validation['valid'] else "WARNING"
        report.append(f"  Dataset {task_id}: {samples} samples [{status}]")
        if not validation['valid']:
            report.append(f"    Issue: {validation.get('error', 'Unknown')}")

    report.append("")
    report.append(f"Total: {total_samples} samples across {len(prediction_files)} datasets")
    report.append("")
    report.append(f"Submission file: {output_path}")
    report.append("=" * 60)

    return '\n'.join(report)


# ============================================================================
# Main Execution
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Package FlagOS predictions for submission"
    )
    parser.add_argument(
        '--predictions_dir', type=str,
        default='../outputs',
        help='Directory with prediction files'
    )
    parser.add_argument(
        '--output', type=str,
        default=None,
        help='Output ZIP file path (default: submission_TIMESTAMP.zip)'
    )
    parser.add_argument(
        '--skip_validation', action='store_true',
        help='Skip file validation'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("FlagOS Submission Packaging")
    print("=" * 60)

    # Find prediction files
    print(f"\nScanning: {args.predictions_dir}")
    prediction_files = find_prediction_files(args.predictions_dir)

    if not prediction_files:
        print("No prediction files found!")
        print(f"Please run annotate.py first to generate predictions in {args.predictions_dir}")
        return

    print(f"\nFound {len(prediction_files)} prediction files:")

    # Validate files
    all_valid = True
    for task_id, filepath in sorted(prediction_files.items()):
        validation = validate_prediction_file(filepath)
        status = "OK" if validation['valid'] else "WARNING"
        print(f"  Dataset {task_id}: {os.path.basename(filepath)} [{status}]")

        if not validation['valid']:
            print(f"    Issue: {validation.get('error', 'Unknown')}")
            all_valid = False

    if not all_valid and not args.skip_validation:
        print("\nSome files have issues. Fix them or use --skip_validation to proceed anyway.")

    # Create output filename
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'submission_{timestamp}.zip'

    # Create ZIP
    print(f"\nCreating submission package: {output_path}")
    create_submission_zip(prediction_files, output_path)

    # Generate report
    report = generate_submission_report(prediction_files, output_path)
    report_path = output_path.replace('.zip', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print("\nSubmission ready!")
    print(f"\nUpload instructions:")
    print(f"1. Go to: https://flagos.io/RaceDetail?id=296fmsd8")
    print(f"2. Upload the ZIP file: {output_path}")
    print(f"3. Check the leaderboard for your score")


if __name__ == "__main__":
    main()
