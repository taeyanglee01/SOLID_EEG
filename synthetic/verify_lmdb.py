"""
Verify LMDB database created from synthetic EEG data.

Usage:
    python verify_lmdb.py --lmdb_path /path/to/database.lmdb
"""

import argparse
import lmdb
import pickle
import numpy as np
from collections import defaultdict


def verify_lmdb(lmdb_path, n_samples_to_check=5):
    """
    Verify LMDB database integrity and content.

    Parameters:
    -----------
    lmdb_path : str
        Path to LMDB database
    n_samples_to_check : int
        Number of samples to inspect in detail
    """
    print("=" * 80)
    print("LMDB Database Verification")
    print("=" * 80)
    print(f"Database path: {lmdb_path}")
    print("")

    # Open database
    try:
        db = lmdb.open(lmdb_path, readonly=True, lock=False)
    except Exception as e:
        print(f"ERROR: Cannot open database: {e}")
        return False

    txn = db.begin()

    # Load keys
    try:
        keys_data = txn.get('__keys__'.encode())
        if keys_data is None:
            print("ERROR: No __keys__ found in database!")
            return False

        keys = pickle.loads(keys_data)
        print(f"Total samples in database: {len(keys)}")
    except Exception as e:
        print(f"ERROR: Cannot load keys: {e}")
        return False

    # Analyze keys
    print("\nKey analysis:")
    task_counts = defaultdict(int)
    subject_counts = defaultdict(int)

    for key in keys:
        # Parse key format: {subject_id}_{task}-{index}
        parts = key.split('_')
        if len(parts) >= 2:
            subject_id = parts[0]
            task = '_'.join(parts[1:]).split('-')[0]

            subject_counts[subject_id] += 1
            task_counts[task] += 1

    print(f"\nNumber of subjects: {len(subject_counts)}")
    print(f"Subjects: {sorted(subject_counts.keys())[:10]}..." if len(subject_counts) > 10
          else f"Subjects: {sorted(subject_counts.keys())}")

    print(f"\nSamples per task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    print(f"\nSamples per subject (first 10):")
    for subject_id, count in sorted(subject_counts.items())[:10]:
        print(f"  {subject_id}: {count}")

    # Check sample data integrity
    print("\n" + "=" * 80)
    print("Sample Data Verification")
    print("=" * 80)

    issues = []

    for i, key in enumerate(keys[:n_samples_to_check]):
        print(f"\nSample {i+1}/{n_samples_to_check}: {key}")

        try:
            data_bytes = txn.get(key.encode())
            if data_bytes is None:
                issues.append(f"Key {key} has no data")
                continue

            data_dict = pickle.loads(data_bytes)

            # Check required fields
            required_fields = ['sample', 'label', 'data_info']
            for field in required_fields:
                if field not in data_dict:
                    issues.append(f"Key {key} missing field: {field}")

            # Check sample data
            sample = data_dict['sample']
            print(f"  Sample shape: {sample.shape}")
            print(f"  Sample dtype: {sample.dtype}")
            print(f"  Sample range: [{sample.min():.2f}, {sample.max():.2f}] µV")
            print(f"  Label: {data_dict['label']}")

            # Verify expected shape (n_channels, 4, 200)
            expected_shape = (64, 4, 200)  # Adjust if needed
            if sample.shape[1:] != expected_shape[1:]:
                print(f"  WARNING: Expected shape (*, 4, 200), got {sample.shape}")

            # Check for NaN or Inf
            if np.any(np.isnan(sample)):
                issues.append(f"Key {key} contains NaN values")
            if np.any(np.isinf(sample)):
                issues.append(f"Key {key} contains Inf values")

            # Check data_info
            data_info = data_dict['data_info']
            print(f"  Dataset: {data_info.get('Dataset')}")
            print(f"  Subject: {data_info.get('subject_id')}")
            print(f"  Task: {data_info.get('task')}")
            print(f"  Resampling rate: {data_info.get('resampling_rate')} Hz")
            print(f"  Number of channels: {len(data_info.get('channel_names', []))}")

            # Check xyz coordinates
            xyz = data_info.get('xyz_id')
            if xyz is not None:
                print(f"  XYZ coordinates shape: {xyz.shape}")
                if np.any(np.isnan(xyz)):
                    print(f"  WARNING: XYZ contains NaN values")

        except Exception as e:
            issues.append(f"Error processing key {key}: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)

    if len(issues) == 0:
        print("✓ All checks passed!")
        print(f"  - {len(keys)} samples found")
        print(f"  - {len(subject_counts)} subjects")
        print(f"  - {len(task_counts)} different tasks")
        print(f"  - No data integrity issues")
    else:
        print("✗ Issues found:")
        for issue in issues:
            print(f"  - {issue}")

    # Database statistics
    print("\n" + "=" * 80)
    print("Database Statistics")
    print("=" * 80)

    stat = db.stat()
    print(f"  Page size: {stat['psize']} bytes")
    print(f"  Depth: {stat['depth']}")
    print(f"  Branch pages: {stat['branch_pages']}")
    print(f"  Leaf pages: {stat['leaf_pages']}")
    print(f"  Overflow pages: {stat['overflow_pages']}")
    print(f"  Entries: {stat['entries']}")

    db.close()

    print("\n" + "=" * 80)

    return len(issues) == 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Verify LMDB database integrity'
    )

    parser.add_argument(
        '--lmdb_path',
        type=str,
        required=True,
        help='Path to LMDB database'
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=5,
        help='Number of samples to inspect in detail (default: 5)'
    )

    args = parser.parse_args()

    success = verify_lmdb(args.lmdb_path, n_samples_to_check=args.n_samples)

    if success:
        print("\n✓ Database verification PASSED")
        exit(0)
    else:
        print("\n✗ Database verification FAILED")
        exit(1)


if __name__ == "__main__":
    main()
