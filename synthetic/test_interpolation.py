"""
Test script for evaluating EEG spatial interpolation methods.

This script loads synthetic EEG data with bad channels and compares
interpolation results against ground truth.
"""

import numpy as np
import mne
from pathlib import Path
import json


def calculate_metrics(ground_truth, interpolated, channel_names):
    """
    Calculate interpolation quality metrics.

    Parameters:
    -----------
    ground_truth : ndarray
        Ground truth data (n_channels, n_samples)
    interpolated : ndarray
        Interpolated data (n_channels, n_samples)
    channel_names : list
        Names of channels being evaluated

    Returns:
    --------
    metrics : dict
        Dictionary containing various metrics
    """
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((ground_truth - interpolated) ** 2))

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(ground_truth - interpolated))

    # Correlation coefficient
    # Flatten arrays for correlation calculation
    gt_flat = ground_truth.flatten()
    interp_flat = interpolated.flatten()
    correlation = np.corrcoef(gt_flat, interp_flat)[0, 1]

    # Relative Error
    relative_error = np.mean(np.abs(ground_truth - interpolated) / (np.abs(ground_truth) + 1e-10))

    # Signal-to-Noise Ratio (SNR) of interpolation
    signal_power = np.mean(ground_truth ** 2)
    noise_power = np.mean((ground_truth - interpolated) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Per-channel metrics
    per_channel_rmse = np.sqrt(np.mean((ground_truth - interpolated) ** 2, axis=1))
    per_channel_corr = np.array([
        np.corrcoef(ground_truth[i, :], interpolated[i, :])[0, 1]
        for i in range(ground_truth.shape[0])
    ])

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'relative_error': relative_error,
        'snr_db': snr_db,
        'per_channel_rmse': dict(zip(channel_names, per_channel_rmse)),
        'per_channel_correlation': dict(zip(channel_names, per_channel_corr))
    }

    return metrics


def test_mne_interpolation(data_path, scenario='default'):
    """
    Test MNE's built-in interpolation method.

    Parameters:
    -----------
    data_path : str or Path
        Path to the synthetic data directory
    scenario : str
        Scenario name to test

    Returns:
    --------
    metrics : dict
        Interpolation quality metrics
    """
    data_path = Path(data_path)

    print(f"\nTesting MNE interpolation on {scenario} scenario...")

    # Load data with bad channels
    raw_file = data_path / f"synthetic_eeg_{scenario}_raw.fif"
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

    bad_channels = raw.info['bads'].copy()
    print(f"Bad channels: {bad_channels}")

    if len(bad_channels) == 0:
        print("No bad channels to interpolate!")
        return None

    # Load ground truth
    gt_file = data_path / f"synthetic_eeg_{scenario}_groundtruth_raw.fif"
    raw_gt = mne.io.read_raw_fif(gt_file, preload=True, verbose=False)

    # Get ground truth data for bad channels
    gt_data = raw_gt.get_data(picks=bad_channels)

    # Perform interpolation
    raw_interp = raw.copy()
    raw_interp.interpolate_bads(reset_bads=False, verbose=False)

    # Get interpolated data
    interp_data = raw_interp.get_data(picks=bad_channels)

    # Calculate metrics
    metrics = calculate_metrics(gt_data, interp_data, bad_channels)

    # Print results
    print(f"\nResults for {scenario}:")
    print(f"  RMSE: {metrics['rmse']:.6e}")
    print(f"  MAE: {metrics['mae']:.6e}")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  Relative Error: {metrics['relative_error']:.4f}")
    print(f"  SNR (dB): {metrics['snr_db']:.2f}")

    print(f"\nPer-channel RMSE:")
    for ch, rmse_val in metrics['per_channel_rmse'].items():
        print(f"  {ch}: {rmse_val:.6e}")

    return metrics


def test_custom_interpolation(data_path, interpolation_func, scenario='default', method_name='custom'):
    """
    Test a custom interpolation method.

    Parameters:
    -----------
    data_path : str or Path
        Path to the synthetic data directory
    interpolation_func : callable
        Function that takes a Raw object and returns interpolated Raw object
        Should have signature: interpolated_raw = interpolation_func(raw)
    scenario : str
        Scenario name to test
    method_name : str
        Name of the interpolation method for display

    Returns:
    --------
    metrics : dict
        Interpolation quality metrics
    """
    data_path = Path(data_path)

    print(f"\nTesting {method_name} interpolation on {scenario} scenario...")

    # Load data with bad channels
    raw_file = data_path / f"synthetic_eeg_{scenario}_raw.fif"
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

    bad_channels = raw.info['bads'].copy()
    print(f"Bad channels: {bad_channels}")

    if len(bad_channels) == 0:
        print("No bad channels to interpolate!")
        return None

    # Load ground truth
    gt_file = data_path / f"synthetic_eeg_{scenario}_groundtruth_raw.fif"
    raw_gt = mne.io.read_raw_fif(gt_file, preload=True, verbose=False)

    # Get ground truth data for bad channels
    gt_data = raw_gt.get_data(picks=bad_channels)

    # Perform custom interpolation
    raw_interp = interpolation_func(raw.copy())

    # Get interpolated data
    interp_data = raw_interp.get_data(picks=bad_channels)

    # Calculate metrics
    metrics = calculate_metrics(gt_data, interp_data, bad_channels)

    # Print results
    print(f"\nResults for {scenario} using {method_name}:")
    print(f"  RMSE: {metrics['rmse']:.6e}")
    print(f"  MAE: {metrics['mae']:.6e}")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  Relative Error: {metrics['relative_error']:.4f}")
    print(f"  SNR (dB): {metrics['snr_db']:.2f}")

    return metrics


def test_spatial_patterns(data_path):
    """
    Test interpolation on spatial pattern validation dataset.

    Parameters:
    -----------
    data_path : str or Path
        Path to the validation data directory
    """
    data_path = Path(data_path)

    print("\n" + "=" * 70)
    print("Testing Spatial Pattern Validation Dataset")
    print("=" * 70)

    # Load ground truth
    gt_file = data_path / "spatial_pattern_groundtruth_raw.fif"
    raw_gt = mne.io.read_raw_fif(gt_file, preload=True, verbose=False)

    # Test different bad channel patterns
    patterns = ['random', 'cluster', 'frontal', 'temporal']

    all_results = {}

    for pattern in patterns:
        test_file = data_path / f"spatial_pattern_{pattern}_raw.fif"

        if not test_file.exists():
            print(f"Skipping {pattern}: file not found")
            continue

        print(f"\n--- Testing {pattern} pattern ---")

        # Load test data
        raw_test = mne.io.read_raw_fif(test_file, preload=True, verbose=False)
        bad_channels = raw_test.info['bads'].copy()

        print(f"Bad channels ({len(bad_channels)}): {bad_channels}")

        # Get ground truth for bad channels
        gt_data = raw_gt.get_data(picks=bad_channels)

        # Interpolate
        raw_interp = raw_test.copy()
        raw_interp.interpolate_bads(reset_bads=False, verbose=False)

        # Get interpolated data
        interp_data = raw_interp.get_data(picks=bad_channels)

        # Calculate metrics
        metrics = calculate_metrics(gt_data, interp_data, bad_channels)

        print(f"  RMSE: {metrics['rmse']:.6e}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  SNR (dB): {metrics['snr_db']:.2f}")

        all_results[pattern] = metrics

    return all_results


def compare_methods(data_path, methods_dict, scenario='default'):
    """
    Compare multiple interpolation methods.

    Parameters:
    -----------
    data_path : str or Path
        Path to the synthetic data directory
    methods_dict : dict
        Dictionary mapping method names to interpolation functions
        e.g., {'MNE': mne_interpolation_func, 'Custom': custom_func}
    scenario : str
        Scenario name to test

    Returns:
    --------
    comparison : dict
        Comparison of all methods
    """
    data_path = Path(data_path)

    print("\n" + "=" * 70)
    print(f"Comparing Interpolation Methods on {scenario} scenario")
    print("=" * 70)

    comparison = {}

    for method_name, interp_func in methods_dict.items():
        metrics = test_custom_interpolation(
            data_path, interp_func, scenario, method_name
        )
        comparison[method_name] = metrics

    # Print comparison table
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Method':<20} {'RMSE':<15} {'Correlation':<15} {'SNR (dB)':<10}")
    print("-" * 70)

    for method_name, metrics in comparison.items():
        if metrics is not None:
            print(f"{method_name:<20} {metrics['rmse']:<15.6e} "
                  f"{metrics['correlation']:<15.4f} {metrics['snr_db']:<10.2f}")

    return comparison


def main():
    """Main function to run all tests."""

    # Set data path
    data_path = Path("C:/Users/Public/Projects/SOLID_EEG/synthetic_data")

    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
        print("Please run generate_synthetic_eeg.py first to create synthetic data.")
        return

    print("=" * 70)
    print("EEG Spatial Interpolation Testing")
    print("=" * 70)

    # Test MNE interpolation on different scenarios
    scenarios = ['default', 'high_snr', 'low_snr', 'event_related']

    print("\n" + "=" * 70)
    print("Testing MNE Built-in Interpolation")
    print("=" * 70)

    results = {}
    for scenario in scenarios:
        scenario_file = data_path / f"synthetic_eeg_{scenario}_raw.fif"
        if scenario_file.exists():
            results[scenario] = test_mne_interpolation(data_path, scenario)
        else:
            print(f"\nSkipping {scenario}: file not found")

    # Test spatial patterns
    validation_path = data_path / "validation"
    if validation_path.exists():
        spatial_results = test_spatial_patterns(validation_path)
    else:
        print("\nValidation data not found. Skipping spatial pattern tests.")

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


# Example of how to use with custom interpolation method
def example_custom_interpolation(raw):
    """
    Example custom interpolation function.

    Replace this with your own interpolation algorithm.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object with bad channels marked

    Returns:
    --------
    raw_interpolated : mne.io.Raw
        Raw object with bad channels interpolated
    """
    # For now, just use MNE's interpolation
    # Replace this with your custom method
    raw_interpolated = raw.copy()
    raw_interpolated.interpolate_bads(reset_bads=False, verbose=False)

    return raw_interpolated


if __name__ == "__main__":
    main()

    # Example of comparing multiple methods:
    # Uncomment and modify this section to compare your custom methods

    # data_path = Path("C:/Users/Public/Projects/SOLID_EEG/synthetic_data")
    #
    # methods = {
    #     'MNE': lambda raw: raw.interpolate_bads(reset_bads=False),
    #     'Custom': example_custom_interpolation,
    #     # Add your methods here:
    #     # 'YourMethod': your_interpolation_function,
    # }
    #
    # compare_methods(data_path, methods, scenario='default')
