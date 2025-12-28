"""
Generate synthetic EEG data for testing spatial interpolation models.

This script creates realistic synthetic EEG data with various signal patterns
and saves them in multiple formats for testing interpolation algorithms.
"""

import numpy as np
import mne
from mne.simulation import simulate_raw
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def create_montage_and_info(n_channels=64, sfreq=250):
    """
    Create EEG montage and info structure.

    Parameters:
    -----------
    n_channels : int
        Number of EEG channels (default: 64)
    sfreq : float
        Sampling frequency in Hz (default: 250)

    Returns:
    --------
    info : mne.Info
        Info structure with channel locations
    """
    # Use standard 10-20 system montage
    montage = mne.channels.make_standard_montage('standard_1020')

    # Get channel names (use first n_channels)
    ch_names = montage.ch_names[:n_channels]

    # Create info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    return info

def generate_sinusoidal_sources(n_sources=5, duration=60, sfreq=250):
    """
    Generate sinusoidal source signals with different frequencies.

    Parameters:
    -----------
    n_sources : int
        Number of source signals
    duration : float
        Duration in seconds
    sfreq : float
        Sampling frequency in Hz

    Returns:
    --------
    sources : ndarray
        Source signals array (n_sources, n_samples)
    """
    n_samples = int(duration * sfreq)
    times = np.arange(n_samples) / sfreq

    sources = np.zeros((n_sources, n_samples))

    # Create sources with different frequency bands
    frequencies = [1, 4, 8, 13, 30]  # Delta, Theta, Alpha, Beta, Gamma
    amplitudes = [10, 15, 20, 12, 5]  # Different amplitudes

    for i in range(n_sources):
        freq = frequencies[i % len(frequencies)]
        amp = amplitudes[i % len(amplitudes)]

        # Add main frequency component
        sources[i] = amp * np.sin(2 * np.pi * freq * times)

        # Add harmonics for more realistic signals
        sources[i] += (amp * 0.3) * np.sin(2 * np.pi * 2 * freq * times)

        # Add some phase variation
        phase_mod = 0.5 * np.sin(2 * np.pi * 0.1 * times)
        sources[i] += amp * 0.2 * np.sin(2 * np.pi * freq * times + phase_mod)

    return sources

def create_forward_solution(info, n_sources=5):
    """
    Create a simple forward solution (mixing matrix).

    Parameters:
    -----------
    info : mne.Info
        EEG info structure
    n_sources : int
        Number of sources

    Returns:
    --------
    fwd_matrix : ndarray
        Forward solution matrix (n_channels, n_sources)
    """
    n_channels = len(info['ch_names'])

    # Create random but smooth forward solution
    # Each source projects to multiple channels with spatial smoothness
    fwd_matrix = np.random.randn(n_channels, n_sources)

    # Add spatial structure (neighboring channels have similar projections)
    from scipy.ndimage import gaussian_filter1d
    for i in range(n_sources):
        fwd_matrix[:, i] = gaussian_filter1d(fwd_matrix[:, i], sigma=2)

    # Normalize
    fwd_matrix = fwd_matrix / np.max(np.abs(fwd_matrix))

    return fwd_matrix

def generate_synthetic_eeg(output_dir, scenario='default'):
    """
    Generate synthetic EEG data with different scenarios.

    Parameters:
    -----------
    output_dir : str or Path
        Directory to save output files
    scenario : str
        Scenario type: 'default', 'high_snr', 'low_snr', 'event_related'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    n_channels = 64
    sfreq = 250  # Hz
    duration = 60  # seconds
    n_sources = 5

    print(f"\nGenerating {scenario} scenario...")
    print(f"Channels: {n_channels}, Sampling rate: {sfreq} Hz, Duration: {duration}s")

    # Create info structure
    info = create_montage_and_info(n_channels=n_channels, sfreq=sfreq)

    # Generate source signals
    sources = generate_sinusoidal_sources(n_sources=n_sources, duration=duration, sfreq=sfreq)

    # Create forward solution
    fwd_matrix = create_forward_solution(info, n_sources=n_sources)

    # Project sources to sensors
    eeg_data = fwd_matrix @ sources

    # Add noise based on scenario
    noise_levels = {
        'default': 0.1,
        'high_snr': 0.01,
        'low_snr': 0.5,
        'event_related': 0.1
    }

    noise_level = noise_levels.get(scenario, 0.1)
    noise = noise_level * np.random.randn(*eeg_data.shape)
    eeg_data += noise

    # For event-related scenario, add some sharp events
    if scenario == 'event_related':
        n_events = 20
        event_samples = np.random.randint(sfreq, eeg_data.shape[1] - sfreq, n_events)
        for event_sample in event_samples:
            # Add event-related potential
            erp = create_erp_template(sfreq)
            event_end = min(event_sample + len(erp), eeg_data.shape[1])
            erp_len = event_end - event_sample
            # Add to random subset of channels
            channels_to_add = np.random.choice(n_channels, size=n_channels//2, replace=False)
            eeg_data[channels_to_add, event_sample:event_end] += erp[:erp_len] * 20

    # Create Raw object
    raw = mne.io.RawArray(eeg_data * 1e-6, info)  # Convert to V

    # Mark some channels as bad for interpolation testing
    n_bad_channels = np.random.randint(3, 8)
    bad_channels = np.random.choice(info['ch_names'], size=n_bad_channels, replace=False)
    raw.info['bads'] = bad_channels.tolist()

    print(f"Marked {n_bad_channels} channels as bad: {bad_channels.tolist()}")

    # Save in multiple formats
    # 1. FIF format (MNE native)
    fif_filename = output_dir / f"synthetic_eeg_{scenario}_raw.fif"
    raw.save(fif_filename, overwrite=True)
    print(f"Saved FIF: {fif_filename}")

    # 2. EDF format
    edf_filename = output_dir / f"synthetic_eeg_{scenario}.edf"
    mne.export.export_raw(edf_filename, raw, overwrite=True)
    print(f"Saved EDF: {edf_filename}")

    # 3. Save ground truth data (without bad channels) separately
    raw_clean = raw.copy()
    raw_clean.info['bads'] = []
    clean_filename = output_dir / f"synthetic_eeg_{scenario}_groundtruth_raw.fif"
    raw_clean.save(clean_filename, overwrite=True)
    print(f"Saved ground truth: {clean_filename}")

    # 4. Save metadata
    metadata = {
        'scenario': scenario,
        'n_channels': n_channels,
        'sfreq': sfreq,
        'duration': duration,
        'bad_channels': bad_channels.tolist(),
        'noise_level': noise_level,
        'n_sources': n_sources
    }

    import json
    metadata_filename = output_dir / f"synthetic_eeg_{scenario}_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_filename}")

    return raw

def create_erp_template(sfreq, duration=0.5):
    """
    Create a template event-related potential.

    Parameters:
    -----------
    sfreq : float
        Sampling frequency
    duration : float
        Duration in seconds

    Returns:
    --------
    erp : ndarray
        ERP template
    """
    n_samples = int(duration * sfreq)
    times = np.arange(n_samples) / sfreq

    # Create P300-like waveform
    # Negative peak around 100ms, positive peak around 300ms
    erp = np.zeros(n_samples)

    # N100 component
    n100_latency = 0.1
    n100_width = 0.03
    erp -= 5 * np.exp(-((times - n100_latency) ** 2) / (2 * n100_width ** 2))

    # P300 component
    p300_latency = 0.3
    p300_width = 0.08
    erp += 10 * np.exp(-((times - p300_latency) ** 2) / (2 * p300_width ** 2))

    return erp

def generate_spatial_pattern_dataset(output_dir):
    """
    Generate dataset with known spatial patterns for interpolation validation.

    This creates data where specific spatial patterns are known,
    making it easier to validate interpolation accuracy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating spatial pattern validation dataset...")

    # Parameters
    n_channels = 64
    sfreq = 250
    duration = 10

    info = create_montage_and_info(n_channels=n_channels, sfreq=sfreq)

    # Get electrode positions
    montage = info.get_montage()
    pos = montage.get_positions()['ch_pos']
    ch_names = info['ch_names']

    # Extract coordinates
    coords = np.array([pos[ch] for ch in ch_names])

    # Create spatial patterns
    n_samples = int(duration * sfreq)
    times = np.arange(n_samples) / sfreq

    # Pattern 1: Left-right gradient
    lr_gradient = coords[:, 0]  # X coordinate
    lr_gradient = (lr_gradient - lr_gradient.min()) / (lr_gradient.max() - lr_gradient.min())

    # Pattern 2: Front-back gradient
    fb_gradient = coords[:, 1]  # Y coordinate
    fb_gradient = (fb_gradient - fb_gradient.min()) / (fb_gradient.max() - fb_gradient.min())

    # Pattern 3: Radial pattern from center
    center = coords.mean(axis=0)
    radial = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    radial = (radial - radial.min()) / (radial.max() - radial.min())

    # Create time-varying signals
    eeg_data = np.zeros((n_channels, n_samples))

    for i in range(n_channels):
        # Combine spatial patterns with temporal dynamics
        temporal = np.sin(2 * np.pi * 10 * times)  # 10 Hz

        eeg_data[i] = (
            lr_gradient[i] * temporal +
            fb_gradient[i] * np.sin(2 * np.pi * 5 * times) +
            radial[i] * np.sin(2 * np.pi * 15 * times)
        )

    # Add small noise
    eeg_data += 0.05 * np.random.randn(*eeg_data.shape)

    # Create Raw object
    raw = mne.io.RawArray(eeg_data * 1e-6, info)

    # Save complete data as ground truth
    gt_filename = output_dir / "spatial_pattern_groundtruth_raw.fif"
    raw.save(gt_filename, overwrite=True)
    print(f"Saved ground truth: {gt_filename}")

    # Create versions with different bad channel patterns
    test_patterns = {
        'random': np.random.choice(ch_names, size=5, replace=False),
        'cluster': ch_names[10:15],  # Clustered region
        'frontal': [ch for ch in ch_names if ch.startswith('Fp') or ch.startswith('AF')],
        'temporal': [ch for ch in ch_names if ch.startswith('T') or ch.startswith('TP')]
    }

    for pattern_name, bad_chans in test_patterns.items():
        raw_test = raw.copy()
        raw_test.info['bads'] = list(bad_chans)

        test_filename = output_dir / f"spatial_pattern_{pattern_name}_raw.fif"
        raw_test.save(test_filename, overwrite=True)
        print(f"Saved {pattern_name} pattern: {test_filename} (bad channels: {bad_chans})")

def main():
    """Main function to generate all synthetic datasets."""

    output_dir = Path("/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg")

    print("=" * 70)
    print("Synthetic EEG Data Generation for Spatial Interpolation Testing")
    print("=" * 70)

    # Generate different scenarios
    scenarios = ['default', 'high_snr', 'low_snr', 'event_related']

    for scenario in scenarios:
        try:
            generate_synthetic_eeg(output_dir, scenario=scenario)
        except Exception as e:
            print(f"Error generating {scenario}: {e}")

    # Generate spatial pattern validation dataset
    try:
        generate_spatial_pattern_dataset(output_dir / "validation")
    except Exception as e:
        print(f"Error generating spatial pattern dataset: {e}")

    print("\n" + "=" * 70)
    print("Data generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Print summary
    print("\nGenerated files:")
    if output_dir.exists():
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.relative_to(output_dir)}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
