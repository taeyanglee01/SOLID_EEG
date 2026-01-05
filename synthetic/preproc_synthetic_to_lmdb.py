"""
Preprocess synthetic EEG data and convert to LMDB format.

This script processes multi-subject synthetic EEG data with task-specific
preprocessing and saves to LMDB database compatible with existing models.

Usage:
    python preproc_synthetic_to_lmdb.py \
        --input_dir /path/to/multisubject_data \
        --output_lmdb /path/to/output.lmdb \
        --tasks motor_imagery,seizure,p300,emotion
"""

import os
import argparse
import lmdb
import pickle
import numpy as np
import mne
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

mne.set_log_level('WARNING')


def get_channel_xyz(info):
    """
    Extract channel XYZ coordinates from MNE info.

    Parameters:
    -----------
    info : mne.Info
        MNE info object with montage

    Returns:
    --------
    xyz_array : ndarray
        Channel coordinates (n_channels, 3)
    """
    montage = info.get_montage()
    pos = montage.get_positions()['ch_pos']
    ch_names = info['ch_names']

    xyz_list = []
    for ch in ch_names:
        if ch in pos:
            xyz_list.append(pos[ch])
        else:
            print(f"[Warning] Channel {ch} position not found, using NaN")
            xyz_list.append(np.full(3, np.nan))

    return np.vstack(xyz_list)


def preprocess_raw(raw, target_sfreq=200, apply_car=True, apply_filters=True):
    """
    Apply standard preprocessing to raw EEG.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    target_sfreq : float
        Target sampling frequency for resampling
    apply_car : bool
        Whether to apply common average reference
    apply_filters : bool
        Whether to apply bandpass/notch filters

    Returns:
    --------
    raw : mne.io.Raw
        Preprocessed raw data
    """
    # Interpolate bad channels
    if len(raw.info['bads']) > 0:
        print(f"    Interpolating {len(raw.info['bads'])} bad channels")
        raw.interpolate_bads(reset_bads=True)

    # Common average reference
    if apply_car:
        raw.set_eeg_reference(ref_channels='average', projection=False)

    # Filters (only if data has artifacts/noise)
    if apply_filters:
        # High-pass filter
        raw.filter(l_freq=0.3, h_freq=None, verbose=False)

        # Notch filter (only if line noise was added)
        # raw.notch_filter(60, verbose=False)  # Uncomment if needed

    # Resample
    original_sfreq = raw.info['sfreq']
    if abs(original_sfreq - target_sfreq) > 0.1:
        raw.resample(target_sfreq, verbose=False)

    return raw


def process_motor_imagery(raw, events, subject_id, task_name, epoch_duration=4.0, target_sfreq=200):
    """
    Process motor imagery data into epochs.

    Motor imagery paradigm: events mark imagery onset, epoch 4 seconds.

    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed raw data
    events : ndarray
        Event array (n_events, 3)
    subject_id : str
        Subject ID (e.g., 'sub-001')
    task_name : str
        Task name (e.g., 'left_hand')
    epoch_duration : float
        Epoch duration in seconds
    target_sfreq : float
        Sampling frequency

    Returns:
    --------
    samples_list : list of dict
        List of processed samples ready for LMDB
    """
    samples_list = []

    if events is None or len(events) == 0:
        print(f"    Warning: No events found for {subject_id}/{task_name}")
        return samples_list

    # Create epochs
    event_id = {'imagery': 1}
    tmax = epoch_duration - 1.0 / raw.info['sfreq']

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Get data in µV
    data = epochs.get_data(units='uV')  # (n_epochs, n_channels, n_samples)
    n_epochs, n_channels, n_samples = data.shape

    print(f"    Motor Imagery: {n_epochs} epochs, shape={data.shape}")

    # Take last 4 seconds (should be all of it if epoch_duration=4)
    last_samples = int(epoch_duration * target_sfreq)
    data = data[:, :, -last_samples:]

    # Reshape to (n_epochs, n_channels, 4, 200)
    # Split 4 seconds into 4 x 1-second segments
    bz, ch_nums, _ = data.shape
    data = data.reshape(bz, ch_nums, 4, int(target_sfreq))

    # Get channel info
    ch_names = [ch.upper() for ch in raw.ch_names]
    xyz_array = get_channel_xyz(raw.info)

    # Create samples
    for i, sample in enumerate(data):
        sample_key = f"{subject_id}_motor_imagery_{task_name}-{i}"

        # Label: 0 for all motor imagery (or differentiate by task if needed)
        label = {'left_hand': 0, 'right_hand': 1, 'both_hands': 2, 'feet': 3}.get(task_name, 0)

        data_dict = {
            'sample': sample,  # (n_channels, 4, 200)
            'label': label,
            'data_info': {
                'Dataset': 'Synthetic-MI',
                'modality': 'EEG',
                'release': 'synthetic',
                'subject_id': subject_id,
                'task': f'motor_imagery_{task_name}',
                'resampling_rate': int(target_sfreq),
                'original_sampling_rate': 250,
                'segment_index': i,
                'start_time': i * epoch_duration,
                'channel_names': ch_names,
                'xyz_id': xyz_array
            }
        }

        samples_list.append((sample_key, data_dict))

    return samples_list


def process_seizure(raw, events, subject_id, seizure_type, epoch_duration=4.0, target_sfreq=200):
    """
    Process seizure data.

    For seizure data, we can either:
    1. Epoch around seizure onsets
    2. Use continuous data and mark seizure periods

    Here we epoch around seizure events.
    """
    samples_list = []

    if events is None or len(events) == 0:
        print(f"    Warning: No seizure events found")
        # Use continuous data
        return process_continuous(raw, subject_id, f'seizure_{seizure_type}',
                                 epoch_duration, target_sfreq, label=1)

    # Create epochs around seizure onsets
    # Use longer epochs to capture seizure activity
    seizure_epoch_duration = 10.0  # 10 seconds
    event_id = {'seizure': 2} if events[0, 2] == 2 else {'seizure': 3}
    tmax = seizure_epoch_duration - 1.0 / raw.info['sfreq']

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    data = epochs.get_data(units='uV')
    n_epochs, n_channels, n_samples = data.shape

    print(f"    Seizure: {n_epochs} epochs, shape={data.shape}")

    # Split into 4-second segments
    segment_samples = int(epoch_duration * target_sfreq)
    n_segments_per_epoch = int(seizure_epoch_duration / epoch_duration)

    ch_names = [ch.upper() for ch in raw.ch_names]
    xyz_array = get_channel_xyz(raw.info)

    segment_idx = 0
    for epoch_idx, epoch_data in enumerate(data):
        for seg in range(n_segments_per_epoch):
            start_sample = seg * segment_samples
            end_sample = start_sample + segment_samples

            if end_sample > n_samples:
                break

            segment = epoch_data[:, start_sample:end_sample]
            segment = segment.reshape(n_channels, 4, int(target_sfreq))

            sample_key = f"{subject_id}_seizure_{seizure_type}-{segment_idx}"

            data_dict = {
                'sample': segment,
                'label': 1,  # Seizure label
                'data_info': {
                    'Dataset': 'Synthetic-Seizure',
                    'modality': 'EEG',
                    'release': 'synthetic',
                    'subject_id': subject_id,
                    'task': f'seizure_{seizure_type}',
                    'resampling_rate': int(target_sfreq),
                    'original_sampling_rate': 250,
                    'segment_index': segment_idx,
                    'start_time': epoch_idx * seizure_epoch_duration + seg * epoch_duration,
                    'channel_names': ch_names,
                    'xyz_id': xyz_array
                }
            }

            samples_list.append((sample_key, data_dict))
            segment_idx += 1

    return samples_list


def process_p300(raw, events, subject_id, epoch_duration=1.0, target_sfreq=200):
    """
    Process P300 oddball data.

    Epoch around stimulus presentations (target vs standard).
    """
    samples_list = []

    if events is None or len(events) == 0:
        print(f"    Warning: No P300 events found")
        return samples_list

    # Event IDs: 1=standard, 2=target
    event_id = {'standard': 1, 'target': 2}
    tmax = epoch_duration - 1.0 / raw.info['sfreq']

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    data = epochs.get_data(units='uV')
    event_labels = epochs.events[:, 2]  # 1 or 2

    n_epochs, n_channels, n_samples = data.shape
    print(f"    P300: {n_epochs} epochs, shape={data.shape}")

    # Ensure exactly 1 second of data
    target_samples = int(epoch_duration * target_sfreq)
    data = data[:, :, :target_samples]

    # For P300, we typically don't split into sub-segments
    # But to match the (ch, 4, 200) format, we need to adapt
    # Option 1: Pad to 4 seconds
    # Option 2: Use 1-second epochs and reshape to (ch, 1, 200), then pad

    # Here we'll use 4-second epochs by combining 4 consecutive 1-second epochs
    ch_names = [ch.upper() for ch in raw.ch_names]
    xyz_array = get_channel_xyz(raw.info)

    segment_idx = 0

    # Process only target trials (event_label == 2)
    target_indices = np.where(event_labels == 2)[0]

    for i in target_indices:
        # Take 4 consecutive epochs starting from this target
        if i + 3 < len(data):
            # Combine 4 x 1-second epochs into 1 x 4-second
            combined = np.concatenate([data[i+j] for j in range(4)], axis=1)  # (ch, 800)
            combined = combined.reshape(n_channels, 4, int(target_sfreq))

            sample_key = f"{subject_id}_p300_oddball-{segment_idx}"

            data_dict = {
                'sample': combined,
                'label': 1,  # Target
                'data_info': {
                    'Dataset': 'Synthetic-P300',
                    'modality': 'EEG',
                    'release': 'synthetic',
                    'subject_id': subject_id,
                    'task': 'p300_oddball',
                    'resampling_rate': int(target_sfreq),
                    'original_sampling_rate': 250,
                    'segment_index': segment_idx,
                    'start_time': i * epoch_duration,
                    'channel_names': ch_names,
                    'xyz_id': xyz_array
                }
            }

            samples_list.append((sample_key, data_dict))
            segment_idx += 1

    return samples_list


def process_continuous(raw, subject_id, task_name, epoch_duration=4.0, target_sfreq=200, label=0):
    """
    Process continuous data (for emotion tasks).

    Split continuous data into non-overlapping 4-second segments.
    """
    samples_list = []

    # Get continuous data
    data = raw.get_data(units='uV')  # (n_channels, n_samples)
    n_channels, n_samples = data.shape

    # Calculate number of 4-second segments
    segment_samples = int(epoch_duration * target_sfreq)
    n_segments = n_samples // segment_samples

    print(f"    Continuous: {n_segments} segments of {epoch_duration}s")

    ch_names = [ch.upper() for ch in raw.ch_names]
    xyz_array = get_channel_xyz(raw.info)

    for i in range(n_segments):
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples

        segment = data[:, start_sample:end_sample]
        segment = segment.reshape(n_channels, 4, int(target_sfreq))

        sample_key = f"{subject_id}_{task_name}-{i}"

        data_dict = {
            'sample': segment,
            'label': label,
            'data_info': {
                'Dataset': f'Synthetic-{task_name}',
                'modality': 'EEG',
                'release': 'synthetic',
                'subject_id': subject_id,
                'task': task_name,
                'resampling_rate': int(target_sfreq),
                'original_sampling_rate': 250,
                'segment_index': i,
                'start_time': i * epoch_duration,
                'channel_names': ch_names,
                'xyz_id': xyz_array
            }
        }

        samples_list.append((sample_key, data_dict))

    return samples_list


def process_subject(subject_dir, tasks_config, target_sfreq=200):
    """
    Process all data for one subject.

    Parameters:
    -----------
    subject_dir : Path
        Subject directory
    tasks_config : dict
        Configuration of which tasks to process
    target_sfreq : float
        Target sampling frequency

    Returns:
    --------
    all_samples : list of tuples
        List of (key, data_dict) for LMDB
    stats : dict
        Processing statistics
    """
    subject_id = subject_dir.name
    all_samples = []
    stats = defaultdict(int)

    print(f"\nProcessing {subject_id}...")

    # -------------------------------------------------------------------------
    # Motor Imagery
    # -------------------------------------------------------------------------
    if tasks_config.get('motor_imagery', False):
        for task in tasks_config['motor_imagery_tasks']:
            raw_file = subject_dir / f"motor_imagery_{task}_raw.fif"
            events_file = subject_dir / f"motor_imagery_{task}_events.txt"

            if not raw_file.exists():
                print(f"  Skipping motor_imagery_{task}: file not found")
                continue

            print(f"  Processing motor_imagery_{task}...")
            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
            raw = preprocess_raw(raw, target_sfreq=target_sfreq, apply_car=True, apply_filters=False)

            # Load events
            if events_file.exists():
                events = np.loadtxt(events_file, dtype=int)
                if events.ndim == 1:
                    events = events.reshape(1, -1)
            else:
                events = None

            samples = process_motor_imagery(raw, events, subject_id, task,
                                           epoch_duration=4.0, target_sfreq=target_sfreq)
            all_samples.extend(samples)
            stats[f'motor_imagery_{task}'] = len(samples)

    # -------------------------------------------------------------------------
    # Seizure
    # -------------------------------------------------------------------------
    if tasks_config.get('seizure', False):
        for seizure_type in tasks_config['seizure_types']:
            raw_file = subject_dir / f"seizure_{seizure_type}_raw.fif"
            events_file = subject_dir / f"seizure_{seizure_type}_events.txt"

            if not raw_file.exists():
                print(f"  Skipping seizure_{seizure_type}: file not found")
                continue

            print(f"  Processing seizure_{seizure_type}...")
            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
            raw = preprocess_raw(raw, target_sfreq=target_sfreq, apply_car=True, apply_filters=False)

            # Load events
            if events_file.exists():
                events = np.loadtxt(events_file, dtype=int)
                if events.ndim == 1:
                    events = events.reshape(1, -1)
            else:
                events = None

            samples = process_seizure(raw, events, subject_id, seizure_type,
                                     epoch_duration=4.0, target_sfreq=target_sfreq)
            all_samples.extend(samples)
            stats[f'seizure_{seizure_type}'] = len(samples)

    # -------------------------------------------------------------------------
    # P300
    # -------------------------------------------------------------------------
    if tasks_config.get('p300', False):
        raw_file = subject_dir / "p300_oddball_raw.fif"
        events_file = subject_dir / "p300_oddball_events.txt"

        if raw_file.exists():
            print(f"  Processing p300_oddball...")
            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
            raw = preprocess_raw(raw, target_sfreq=target_sfreq, apply_car=True, apply_filters=False)

            # Load events
            if events_file.exists():
                events = np.loadtxt(events_file, dtype=int)
                if events.ndim == 1:
                    events = events.reshape(1, -1)
            else:
                events = None

            samples = process_p300(raw, events, subject_id,
                                  epoch_duration=1.0, target_sfreq=target_sfreq)
            all_samples.extend(samples)
            stats['p300_oddball'] = len(samples)

    # -------------------------------------------------------------------------
    # Emotion
    # -------------------------------------------------------------------------
    if tasks_config.get('emotion', False):
        for emotion in tasks_config['emotion_types']:
            raw_file = subject_dir / f"emotion_{emotion}_raw.fif"

            if not raw_file.exists():
                print(f"  Skipping emotion_{emotion}: file not found")
                continue

            print(f"  Processing emotion_{emotion}...")
            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
            raw = preprocess_raw(raw, target_sfreq=target_sfreq, apply_car=True, apply_filters=False)

            # Emotion labels
            emotion_label = {'positive': 0, 'negative': 1, 'neutral': 2}.get(emotion, 2)

            samples = process_continuous(raw, subject_id, f'emotion_{emotion}',
                                        epoch_duration=4.0, target_sfreq=target_sfreq,
                                        label=emotion_label)
            all_samples.extend(samples)
            stats[f'emotion_{emotion}'] = len(samples)

    print(f"  Total samples for {subject_id}: {len(all_samples)}")

    return all_samples, stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Preprocess synthetic EEG and convert to LMDB'
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing subject subdirectories'
    )

    parser.add_argument(
        '--output_lmdb',
        type=str,
        required=True,
        help='Output LMDB database path'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='Tasks to process: all, motor_imagery, seizure, p300, emotion, or comma-separated'
    )

    parser.add_argument(
        '--target_sfreq',
        type=float,
        default=200.0,
        help='Target sampling frequency (Hz) for resampling (default: 200)'
    )

    parser.add_argument(
        '--map_size',
        type=int,
        default=50 * 1024**3,  # 50 GB
        help='LMDB map size in bytes (default: 50GB)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Parse tasks
    tasks_config = {}
    if args.tasks == 'all':
        tasks_config = {
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

        if 'motor_imagery' in task_list or 'motor' in task_list:
            tasks_config['motor_imagery'] = True
            tasks_config['motor_imagery_tasks'] = ['left_hand', 'right_hand', 'both_hands', 'feet']

        if 'seizure' in task_list:
            tasks_config['seizure'] = True
            tasks_config['seizure_types'] = ['absence', 'focal_frontal', 'focal_temporal']

        if 'p300' in task_list:
            tasks_config['p300'] = True

        if 'emotion' in task_list:
            tasks_config['emotion'] = True
            tasks_config['emotion_types'] = ['positive', 'negative', 'neutral']

    print("=" * 80)
    print("Synthetic EEG Preprocessing and LMDB Conversion")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output LMDB: {args.output_lmdb}")
    print(f"Target sampling frequency: {args.target_sfreq} Hz")
    print(f"Tasks: {args.tasks}")
    print("=" * 80)

    # Find all subject directories
    subject_dirs = sorted(input_dir.glob("sub-*"))

    if len(subject_dirs) == 0:
        print("ERROR: No subject directories found!")
        return

    print(f"\nFound {len(subject_dirs)} subjects")

    # Open LMDB
    db = lmdb.open(args.output_lmdb, map_size=args.map_size)

    # Process all subjects
    all_keys = []
    global_stats = defaultdict(int)
    task_stats = defaultdict(int)

    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        samples, stats = process_subject(subject_dir, tasks_config, target_sfreq=args.target_sfreq)

        # Write to LMDB
        txn = db.begin(write=True)
        for key, data_dict in samples:
            txn.put(key=key.encode(), value=pickle.dumps(data_dict))
            all_keys.append(key)
        txn.commit()

        # Update statistics
        for task_name, count in stats.items():
            global_stats[task_name] += count
            task_stats[task_name] += count

    # Save keys
    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(all_keys))
    txn.commit()

    db.close()

    # Print statistics
    print("\n" + "=" * 80)
    print("Processing Statistics")
    print("=" * 80)
    print(f"Total subjects processed: {len(subject_dirs)}")
    print(f"Total samples saved: {len(all_keys)}")
    print("\nSamples per task:")
    for task_name, count in sorted(task_stats.items()):
        print(f"  {task_name}: {count}")
    print("=" * 80)

    print(f"\nLMDB database created at: {args.output_lmdb}")
    print("Done!")


if __name__ == "__main__":
    main()
