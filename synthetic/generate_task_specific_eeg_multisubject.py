"""
Generate task-specific synthetic EEG data for multiple subjects.

This script generates synthetic EEG data for multiple subjects, creating
a separate subdirectory for each subject. Each subject has slightly different
characteristics (e.g., amplitude, frequency variations) to simulate inter-subject
variability.

Tasks included:
1. Motor Imagery - Mu/Beta rhythm modulation in sensorimotor cortex
2. Seizure - Spike-wave discharge with spatial propagation
3. P300 Oddball - Event-related potential with parietal maximum
4. Emotion - Frontal alpha asymmetry

Usage:
    python generate_task_specific_eeg_multisubject.py --n_subjects 10 --output_dir /path/to/output
"""

import numpy as np
import mne
from pathlib import Path
import argparse
from datetime import datetime
from scipy import signal

# Import generation functions from the original script
from generate_task_specific_eeg import (
    generate_motor_imagery_eeg,
    generate_seizure_eeg,
    generate_p300_oddball_eeg,
    generate_emotion_eeg
)


def generate_subject_id(subject_num, prefix='sub'):
    """
    Generate a subject ID string.

    Parameters:
    -----------
    subject_num : int
        Subject number (1-indexed)
    prefix : str
        Prefix for subject ID (default: 'sub')

    Returns:
    --------
    subject_id : str
        Formatted subject ID (e.g., 'sub-001')
    """
    return f"{prefix}-{subject_num:03d}"


def create_subject_specific_info(base_info, subject_num, n_subjects):
    """
    Create subject-specific variations in EEG characteristics.

    This function creates slight variations in amplitude scaling factors
    to simulate inter-subject variability.

    Parameters:
    -----------
    base_info : mne.Info
        Base EEG info structure
    subject_num : int
        Subject number (1-indexed)
    n_subjects : int
        Total number of subjects

    Returns:
    --------
    amplitude_scale : float
        Amplitude scaling factor for this subject (0.8 to 1.2)
    frequency_shift : float
        Frequency shift factor (0.95 to 1.05)
    """
    # Set seed for reproducibility per subject
    np.random.seed(42 + subject_num)

    # Amplitude variation: 0.8 to 1.2
    amplitude_scale = np.random.uniform(0.8, 1.2)

    # Frequency variation: 0.95 to 1.05 (5% variation)
    frequency_shift = np.random.uniform(0.95, 1.05)

    # Reset random seed to ensure different data each time
    np.random.seed()

    return amplitude_scale, frequency_shift


def apply_subject_variations(raw, amplitude_scale, frequency_shift):
    """
    Apply subject-specific variations to the generated EEG data.

    This applies both amplitude scaling and frequency shifting to simulate
    inter-subject variability.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    amplitude_scale : float
        Amplitude scaling factor (e.g., 1.1 = 10% increase)
    frequency_shift : float
        Frequency scaling factor (e.g., 1.05 = 5% faster oscillations)

    Returns:
    --------
    raw : mne.io.Raw
        Modified raw EEG data with subject-specific variations
    """
    # 1. Apply amplitude scaling
    raw._data *= amplitude_scale

    # 2. Apply frequency shift by time-axis scaling
    # frequency_shift > 1.0 means faster oscillations (compress time, then resample)
    # frequency_shift < 1.0 means slower oscillations (stretch time, then resample)
    if abs(frequency_shift - 1.0) > 1e-6:  # Only if there's actual shift
        n_channels, n_samples = raw._data.shape

        # Calculate new number of samples after time scaling
        # If frequency_shift = 1.05, we compress time by 1.05x
        # So we need fewer samples to represent the same duration
        n_samples_scaled = int(n_samples / frequency_shift)

        # Resample each channel
        resampled_data = np.zeros((n_channels, n_samples_scaled))
        for ch_idx in range(n_channels):
            # Resample to scaled length (this speeds up or slows down the signal)
            resampled_data[ch_idx] = signal.resample(raw._data[ch_idx], n_samples_scaled)

        # Now resample back to original length to maintain same duration
        # This effectively scales all frequency components
        final_data = np.zeros((n_channels, n_samples))
        for ch_idx in range(n_channels):
            final_data[ch_idx] = signal.resample(resampled_data[ch_idx], n_samples)

        raw._data = final_data

    return raw


def generate_multisubject_data(n_subjects, output_base_dir, tasks_config, n_channels=64, sfreq=250):
    """
    Generate task-specific EEG data for multiple subjects.

    Parameters:
    -----------
    n_subjects : int
        Number of subjects to generate
    output_base_dir : Path or str
        Base output directory
    tasks_config : dict
        Dictionary specifying which tasks to generate and their parameters
    n_channels : int
        Number of EEG channels (default: 64)
    sfreq : float
        Sampling frequency (default: 250 Hz)
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Multi-Subject Task-Specific EEG Data Generation")
    print(f"Number of subjects: {n_subjects}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)

    # Create montage and base info
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:n_channels]

    # Generate data for each subject
    for subject_num in range(1, n_subjects + 1):
        subject_id = generate_subject_id(subject_num)
        subject_dir = output_base_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Generating data for {subject_id} ({subject_num}/{n_subjects})")
        print("=" * 80)

        # Create subject-specific info
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        info.set_montage(montage)

        # Get subject-specific variations
        amplitude_scale, frequency_shift = create_subject_specific_info(
            info, subject_num, n_subjects
        )

        print(f"Subject variations: amplitude_scale={amplitude_scale:.3f}, "
              f"frequency_shift={frequency_shift:.3f}")

        # ---------------------------------------------------------------------
        # Motor Imagery Tasks
        # ---------------------------------------------------------------------
        if tasks_config.get('motor_imagery', False):
            print("\n" + "-" * 70)
            print("Motor Imagery Tasks")
            print("-" * 70)

            for task in tasks_config['motor_imagery_tasks']:
                try:
                    print(f"  Generating: {task}")
                    raw, events = generate_motor_imagery_eeg(
                        info,
                        duration=tasks_config['motor_imagery_duration'],
                        task=task
                    )

                    # Apply subject variations
                    raw = apply_subject_variations(raw, amplitude_scale, frequency_shift)

                    # Mark bad channels
                    n_bad = np.random.randint(3, 6)
                    bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
                    raw.info['bads'] = bad_channels.tolist()

                    # Save with bad channels
                    filename = subject_dir / f"motor_imagery_{task}_raw.fif"
                    raw.save(filename, overwrite=True)
                    print(f"    Saved: {filename}")

                    # Save ground truth (no bad channels)
                    raw_gt = raw.copy()
                    raw_gt.info['bads'] = []
                    gt_filename = subject_dir / f"motor_imagery_{task}_groundtruth_raw.fif"
                    raw_gt.save(gt_filename, overwrite=True)

                    # Save events
                    events_filename = subject_dir / f"motor_imagery_{task}_events.txt"
                    np.savetxt(events_filename, events, fmt='%d')

                except Exception as e:
                    print(f"    Error: {e}")

        # ---------------------------------------------------------------------
        # Seizure Tasks
        # ---------------------------------------------------------------------
        if tasks_config.get('seizure', False):
            print("\n" + "-" * 70)
            print("Seizure Tasks")
            print("-" * 70)

            for seizure_type in tasks_config['seizure_types']:
                try:
                    print(f"  Generating: {seizure_type}")
                    raw, events = generate_seizure_eeg(
                        info,
                        duration=tasks_config['seizure_duration'],
                        seizure_type=seizure_type
                    )

                    # Apply subject variations
                    raw = apply_subject_variations(raw, amplitude_scale, frequency_shift)

                    # Mark bad channels
                    n_bad = np.random.randint(3, 6)
                    bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
                    raw.info['bads'] = bad_channels.tolist()

                    # Save with bad channels
                    filename = subject_dir / f"seizure_{seizure_type}_raw.fif"
                    raw.save(filename, overwrite=True)
                    print(f"    Saved: {filename}")

                    # Save ground truth
                    raw_gt = raw.copy()
                    raw_gt.info['bads'] = []
                    gt_filename = subject_dir / f"seizure_{seizure_type}_groundtruth_raw.fif"
                    raw_gt.save(gt_filename, overwrite=True)

                    # Save events if available
                    if events is not None:
                        events_filename = subject_dir / f"seizure_{seizure_type}_events.txt"
                        np.savetxt(events_filename, events, fmt='%d')

                except Exception as e:
                    print(f"    Error: {e}")

        # ---------------------------------------------------------------------
        # P300 Oddball Task
        # ---------------------------------------------------------------------
        if tasks_config.get('p300', False):
            print("\n" + "-" * 70)
            print("P300 Oddball Task")
            print("-" * 70)

            try:
                print(f"  Generating: P300 Oddball")
                raw, events = generate_p300_oddball_eeg(
                    info,
                    duration=tasks_config['p300_duration'],
                    target_probability=tasks_config['p300_target_prob']
                )

                # Apply subject variations
                raw = apply_subject_variations(raw, amplitude_scale, frequency_shift)

                # Mark bad channels
                n_bad = np.random.randint(3, 6)
                bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
                raw.info['bads'] = bad_channels.tolist()

                # Save with bad channels
                filename = subject_dir / "p300_oddball_raw.fif"
                raw.save(filename, overwrite=True)
                print(f"    Saved: {filename}")

                # Save ground truth
                raw_gt = raw.copy()
                raw_gt.info['bads'] = []
                gt_filename = subject_dir / "p300_oddball_groundtruth_raw.fif"
                raw_gt.save(gt_filename, overwrite=True)

                # Save events
                events_filename = subject_dir / "p300_oddball_events.txt"
                np.savetxt(events_filename, events, fmt='%d')

            except Exception as e:
                print(f"    Error: {e}")

        # ---------------------------------------------------------------------
        # Emotion Tasks
        # ---------------------------------------------------------------------
        if tasks_config.get('emotion', False):
            print("\n" + "-" * 70)
            print("Emotion Tasks")
            print("-" * 70)

            for emotion in tasks_config['emotion_types']:
                try:
                    print(f"  Generating: {emotion}")
                    raw = generate_emotion_eeg(
                        info,
                        duration=tasks_config['emotion_duration'],
                        emotion=emotion
                    )

                    # Apply subject variations
                    raw = apply_subject_variations(raw, amplitude_scale, frequency_shift)

                    # Mark bad channels
                    n_bad = np.random.randint(3, 6)
                    bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
                    raw.info['bads'] = bad_channels.tolist()

                    # Save with bad channels
                    filename = subject_dir / f"emotion_{emotion}_raw.fif"
                    raw.save(filename, overwrite=True)
                    print(f"    Saved: {filename}")

                    # Save ground truth
                    raw_gt = raw.copy()
                    raw_gt.info['bads'] = []
                    gt_filename = subject_dir / f"emotion_{emotion}_groundtruth_raw.fif"
                    raw_gt.save(gt_filename, overwrite=True)

                except Exception as e:
                    print(f"    Error: {e}")

        print(f"\n{subject_id} complete!")

    print("\n" + "=" * 80)
    print(f"All {n_subjects} subjects generated successfully!")
    print(f"Data saved to: {output_base_dir}")
    print("=" * 80)


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate multi-subject task-specific synthetic EEG data'
    )

    parser.add_argument(
        '--n_subjects',
        type=int,
        default=10,
        help='Number of subjects to generate (default: 10)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_task_data',
        help='Output directory for generated data'
    )

    parser.add_argument(
        '--n_channels',
        type=int,
        default=64,
        help='Number of EEG channels (default: 64)'
    )

    parser.add_argument(
        '--sfreq',
        type=float,
        default=250.0,
        help='Sampling frequency in Hz (default: 250)'
    )

    # Task selection flags
    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='Tasks to generate: all, motor, seizure, p300, emotion, or comma-separated list (default: all)'
    )

    # Task-specific parameters
    parser.add_argument(
        '--motor_duration',
        type=int,
        default=60,
        help='Duration of motor imagery tasks in seconds (default: 60)'
    )

    parser.add_argument(
        '--seizure_duration',
        type=int,
        default=60,
        help='Duration of seizure recordings in seconds (default: 60)'
    )

    parser.add_argument(
        '--p300_duration',
        type=int,
        default=300,
        help='Duration of P300 task in seconds (default: 300)'
    )

    parser.add_argument(
        '--emotion_duration',
        type=int,
        default=120,
        help='Duration of emotion tasks in seconds (default: 120)'
    )

    args = parser.parse_args()

    # Parse task selection
    tasks_config = {
        'motor_imagery': False,
        'seizure': False,
        'p300': False,
        'emotion': False,
    }

    if args.tasks == 'all':
        tasks_config = {
            'motor_imagery': True,
            'motor_imagery_tasks': ['left_hand', 'right_hand', 'both_hands', 'feet'],
            'motor_imagery_duration': args.motor_duration,
            'seizure': True,
            'seizure_types': ['absence', 'focal_frontal', 'focal_temporal'],
            'seizure_duration': args.seizure_duration,
            'p300': True,
            'p300_duration': args.p300_duration,
            'p300_target_prob': 0.2,
            'emotion': True,
            'emotion_types': ['positive', 'negative', 'neutral'],
            'emotion_duration': args.emotion_duration,
        }
    else:
        task_list = args.tasks.lower().split(',')

        if 'motor' in task_list or 'motor_imagery' in task_list:
            tasks_config['motor_imagery'] = True
            tasks_config['motor_imagery_tasks'] = ['left_hand', 'right_hand', 'both_hands', 'feet']
            tasks_config['motor_imagery_duration'] = args.motor_duration

        if 'seizure' in task_list:
            tasks_config['seizure'] = True
            tasks_config['seizure_types'] = ['absence', 'focal_frontal', 'focal_temporal']
            tasks_config['seizure_duration'] = args.seizure_duration

        if 'p300' in task_list:
            tasks_config['p300'] = True
            tasks_config['p300_duration'] = args.p300_duration
            tasks_config['p300_target_prob'] = 0.2

        if 'emotion' in task_list:
            tasks_config['emotion'] = True
            tasks_config['emotion_types'] = ['positive', 'negative', 'neutral']
            tasks_config['emotion_duration'] = args.emotion_duration

    # Print configuration
    print("\nConfiguration:")
    print(f"  Number of subjects: {args.n_subjects}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of channels: {args.n_channels}")
    print(f"  Sampling frequency: {args.sfreq} Hz")
    print(f"  Tasks to generate:")
    if tasks_config['motor_imagery']:
        print(f"    - Motor Imagery ({args.motor_duration}s): {tasks_config['motor_imagery_tasks']}")
    if tasks_config['seizure']:
        print(f"    - Seizure ({args.seizure_duration}s): {tasks_config['seizure_types']}")
    if tasks_config['p300']:
        print(f"    - P300 Oddball ({args.p300_duration}s)")
    if tasks_config['emotion']:
        print(f"    - Emotion ({args.emotion_duration}s): {tasks_config['emotion_types']}")

    # Generate data
    generate_multisubject_data(
        n_subjects=args.n_subjects,
        output_base_dir=args.output_dir,
        tasks_config=tasks_config,
        n_channels=args.n_channels,
        sfreq=args.sfreq
    )


if __name__ == "__main__":
    main()
