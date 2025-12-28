"""
Validate multi-subject EEG dataset.

This script checks the integrity and completeness of the generated
multi-subject EEG dataset.
"""

import argparse
from pathlib import Path
import mne
import numpy as np
from collections import defaultdict


def validate_subject_data(subject_dir, expected_tasks, verbose=False):
    """
    Validate data for a single subject.

    Parameters:
    -----------
    subject_dir : Path
        Subject directory
    expected_tasks : dict
        Dictionary of expected tasks
    verbose : bool
        Print detailed information

    Returns:
    --------
    is_valid : bool
        Whether the subject data is valid
    issues : list
        List of issues found
    """
    subject_id = subject_dir.name
    issues = []

    if verbose:
        print(f"\nValidating {subject_id}...")

    # Check if directory exists
    if not subject_dir.exists():
        issues.append(f"Directory does not exist: {subject_dir}")
        return False, issues

    # Expected files based on tasks
    expected_files = []

    if expected_tasks.get('motor_imagery', False):
        for task in expected_tasks['motor_imagery_tasks']:
            expected_files.extend([
                f"motor_imagery_{task}_raw.fif",
                f"motor_imagery_{task}_groundtruth_raw.fif",
                f"motor_imagery_{task}_events.txt"
            ])

    if expected_tasks.get('seizure', False):
        for seizure_type in expected_tasks['seizure_types']:
            expected_files.extend([
                f"seizure_{seizure_type}_raw.fif",
                f"seizure_{seizure_type}_groundtruth_raw.fif",
                f"seizure_{seizure_type}_events.txt"
            ])

    if expected_tasks.get('p300', False):
        expected_files.extend([
            "p300_oddball_raw.fif",
            "p300_oddball_groundtruth_raw.fif",
            "p300_oddball_events.txt"
        ])

    if expected_tasks.get('emotion', False):
        for emotion in expected_tasks['emotion_types']:
            expected_files.extend([
                f"emotion_{emotion}_raw.fif",
                f"emotion_{emotion}_groundtruth_raw.fif"
            ])

    # Check for missing files
    missing_files = []
    for fname in expected_files:
        fpath = subject_dir / fname
        if not fpath.exists():
            missing_files.append(fname)

    if missing_files:
        issues.append(f"Missing files: {missing_files}")

    # Check file integrity for .fif files
    fif_files = list(subject_dir.glob("*.fif"))

    for fif_file in fif_files:
        try:
            raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)

            # Check basic properties
            if raw.info['sfreq'] <= 0:
                issues.append(f"{fif_file.name}: Invalid sampling frequency")

            if len(raw.ch_names) == 0:
                issues.append(f"{fif_file.name}: No channels found")

            # Check for NaN or Inf values
            data = raw.get_data()
            if np.any(np.isnan(data)):
                issues.append(f"{fif_file.name}: Contains NaN values")
            if np.any(np.isinf(data)):
                issues.append(f"{fif_file.name}: Contains Inf values")

            # Check bad channels
            if 'groundtruth' in fif_file.name:
                if len(raw.info['bads']) > 0:
                    issues.append(f"{fif_file.name}: Groundtruth should not have bad channels")
            else:
                if len(raw.info['bads']) == 0:
                    issues.append(f"{fif_file.name}: Raw data should have bad channels")

        except Exception as e:
            issues.append(f"{fif_file.name}: Error reading file - {str(e)}")

    # Check event files
    event_files = list(subject_dir.glob("*_events.txt"))

    for event_file in event_files:
        try:
            events = np.loadtxt(event_file)
            if events.size == 0:
                issues.append(f"{event_file.name}: Empty event file")
            elif events.ndim == 1 and events.size >= 3:
                # Single event
                pass
            elif events.ndim == 2 and events.shape[1] != 3:
                issues.append(f"{event_file.name}: Events should have 3 columns")
        except Exception as e:
            issues.append(f"{event_file.name}: Error reading file - {str(e)}")

    is_valid = len(issues) == 0

    if verbose:
        if is_valid:
            print(f"  ✓ {subject_id} is valid")
        else:
            print(f"  ✗ {subject_id} has issues:")
            for issue in issues:
                print(f"    - {issue}")

    return is_valid, issues


def validate_dataset(dataset_dir, expected_tasks, verbose=False):
    """
    Validate entire multi-subject dataset.

    Parameters:
    -----------
    dataset_dir : Path or str
        Dataset base directory
    expected_tasks : dict
        Dictionary of expected tasks
    verbose : bool
        Print detailed information

    Returns:
    --------
    summary : dict
        Validation summary
    """
    dataset_dir = Path(dataset_dir)

    print("=" * 80)
    print("Multi-Subject EEG Dataset Validation")
    print("=" * 80)
    print(f"Dataset directory: {dataset_dir}")
    print("")

    # Find all subject directories
    subject_dirs = sorted(dataset_dir.glob("sub-*"))

    if len(subject_dirs) == 0:
        print("ERROR: No subject directories found!")
        return None

    print(f"Found {len(subject_dirs)} subject directories")
    print("")

    # Validate each subject
    valid_subjects = []
    invalid_subjects = []
    all_issues = defaultdict(list)

    for subject_dir in subject_dirs:
        is_valid, issues = validate_subject_data(subject_dir, expected_tasks, verbose=verbose)

        if is_valid:
            valid_subjects.append(subject_dir.name)
        else:
            invalid_subjects.append(subject_dir.name)
            all_issues[subject_dir.name] = issues

    # Print summary
    print("")
    print("=" * 80)
    print("Validation Summary")
    print("=" * 80)
    print(f"Total subjects: {len(subject_dirs)}")
    print(f"Valid subjects: {len(valid_subjects)}")
    print(f"Invalid subjects: {len(invalid_subjects)}")
    print("")

    if invalid_subjects:
        print("Invalid subjects:")
        for subject_id in invalid_subjects:
            print(f"  - {subject_id}")
            for issue in all_issues[subject_id]:
                print(f"    * {issue}")
        print("")

    # Check dataset statistics
    if len(valid_subjects) > 0:
        print("Dataset Statistics:")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        total_size_gb = total_size / (1024 * 1024 * 1024)

        print(f"  Total size: {total_size_gb:.2f} GB ({total_size_mb:.1f} MB)")
        print(f"  Average per subject: {total_size_mb / len(subject_dirs):.1f} MB")

        # Count files
        total_files = len(list(dataset_dir.rglob("*.fif")))
        total_events = len(list(dataset_dir.rglob("*_events.txt")))

        print(f"  Total .fif files: {total_files}")
        print(f"  Total event files: {total_events}")
        print("")

    # Overall result
    print("=" * 80)
    if len(invalid_subjects) == 0:
        print("✓ DATASET VALIDATION PASSED")
        print("All subjects have valid data!")
    else:
        print("✗ DATASET VALIDATION FAILED")
        print(f"{len(invalid_subjects)} subject(s) have issues.")
    print("=" * 80)

    return {
        'total_subjects': len(subject_dirs),
        'valid_subjects': len(valid_subjects),
        'invalid_subjects': len(invalid_subjects),
        'issues': dict(all_issues),
        'total_size_gb': total_size_gb if len(valid_subjects) > 0 else 0
    }


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Validate multi-subject EEG dataset'
    )

    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='Tasks that were generated: all, motor, seizure, p300, emotion, or comma-separated'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed validation information for each subject'
    )

    args = parser.parse_args()

    # Parse expected tasks
    expected_tasks = {}

    if args.tasks == 'all':
        expected_tasks = {
            'motor_imagery': True,
            'motor_imagery_tasks': ['left_hand', 'right_hand', 'both_hands', 'feet'],
            'seizure': True,
            'seizure_types': ['absence', 'focal_frontal', 'focal_temporal'],
            'p300': True,
            'emotion': True,
            'emotion_types': ['positive', 'negative', 'neutral'],
        }
    else:
        task_list = args.tasks.lower().split(',')

        if 'motor' in task_list or 'motor_imagery' in task_list:
            expected_tasks['motor_imagery'] = True
            expected_tasks['motor_imagery_tasks'] = ['left_hand', 'right_hand', 'both_hands', 'feet']

        if 'seizure' in task_list:
            expected_tasks['seizure'] = True
            expected_tasks['seizure_types'] = ['absence', 'focal_frontal', 'focal_temporal']

        if 'p300' in task_list:
            expected_tasks['p300'] = True

        if 'emotion' in task_list:
            expected_tasks['emotion'] = True
            expected_tasks['emotion_types'] = ['positive', 'negative', 'neutral']

    # Validate dataset
    summary = validate_dataset(
        args.dataset_dir,
        expected_tasks,
        verbose=args.verbose
    )

    # Exit with appropriate code
    if summary and summary['invalid_subjects'] == 0:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
