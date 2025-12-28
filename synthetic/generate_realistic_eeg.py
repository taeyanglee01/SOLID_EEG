"""
Generate more realistic synthetic EEG data for spatial interpolation testing.

This version includes:
- 1/f noise (pink noise)
- Realistic artifacts (EOG, EMG, ECG)
- Non-stationary signals
- Temporal autocorrelation
- Realistic spatial correlations
"""

import numpy as np
import mne
from pathlib import Path
from scipy import signal, ndimage
from scipy.spatial.distance import cdist


def generate_pink_noise(n_samples, alpha=1.5):
    """
    Generate 1/f^alpha noise (pink noise).

    Real EEG has power spectral density P(f) ∝ 1/f^α where α ≈ 1-2

    Parameters:
    -----------
    n_samples : int
        Number of samples
    alpha : float
        Exponent for 1/f^alpha (default: 1.5)

    Returns:
    --------
    pink_noise : ndarray
        Pink noise signal
    """
    # Generate white noise in frequency domain
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # Avoid division by zero

    # Create 1/f^alpha spectrum
    spectrum = 1 / (freqs ** (alpha / 2))

    # Random phase
    phases = np.random.uniform(0, 2*np.pi, len(freqs))

    # Combine magnitude and phase
    fft_signal = spectrum * np.exp(1j * phases)

    # Transform back to time domain
    pink_noise = np.fft.irfft(fft_signal, n=n_samples)

    # Normalize
    pink_noise = pink_noise / np.std(pink_noise)

    return pink_noise


def generate_burst_oscillation(freq, duration, sfreq, burst_rate=0.3):
    """
    Generate burst-like oscillation (like real alpha rhythm).

    Parameters:
    -----------
    freq : float
        Center frequency (Hz)
    duration : float
        Duration (seconds)
    sfreq : float
        Sampling frequency (Hz)
    burst_rate : float
        Proportion of time with bursts (0-1)

    Returns:
    --------
    burst_signal : ndarray
        Burst oscillation
    """
    n_samples = int(duration * sfreq)
    times = np.arange(n_samples) / sfreq

    # Generate base oscillation
    oscillation = np.sin(2 * np.pi * freq * times)

    # Generate amplitude envelope with bursts
    # Use low-pass filtered noise to create smooth bursts
    envelope = np.random.randn(n_samples)

    # Low-pass filter for smooth amplitude changes
    from scipy.signal import butter, filtfilt
    b, a = butter(3, 2 / (sfreq / 2), btype='low')
    envelope = filtfilt(b, a, envelope)

    # Threshold to create bursts
    threshold = np.percentile(envelope, (1 - burst_rate) * 100)
    envelope = np.maximum(envelope - threshold, 0)
    envelope = envelope / (np.max(envelope) + 1e-10)

    # Add random phase resets at burst onsets
    phase_resets = np.diff(envelope > 0.1)
    phase_resets = np.concatenate([[0], phase_resets])

    burst_signal = oscillation * envelope

    return burst_signal


def generate_eog_artifact(n_samples, sfreq, n_blinks=None):
    """
    Generate eye blink artifacts (EOG).

    Eye blinks are large (100-200µV), brief (~400ms), mainly in frontal channels.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    sfreq : float
        Sampling frequency (Hz)
    n_blinks : int or None
        Number of blinks (if None, random between 5-20)

    Returns:
    --------
    eog : ndarray
        EOG artifact signal
    """
    if n_blinks is None:
        n_blinks = np.random.randint(5, 20)

    eog = np.zeros(n_samples)

    # Blink template (biphasic waveform)
    blink_duration = 0.4  # seconds
    blink_samples = int(blink_duration * sfreq)
    blink_times = np.linspace(0, blink_duration, blink_samples)

    # Biphasic template (negative then positive)
    blink_template = (
        -np.exp(-((blink_times - 0.1) ** 2) / (2 * 0.03 ** 2)) +
        0.5 * np.exp(-((blink_times - 0.25) ** 2) / (2 * 0.05 ** 2))
    )

    # Random blink times (avoid edges)
    blink_positions = np.random.randint(
        int(sfreq), n_samples - int(sfreq), n_blinks
    )

    # Add blinks
    for pos in blink_positions:
        end_pos = min(pos + blink_samples, n_samples)
        template_end = end_pos - pos
        eog[pos:end_pos] += blink_template[:template_end]

    # Large amplitude (100-200µV, while EEG is ~10-50µV)
    eog = eog * np.random.uniform(100, 200)

    return eog


def generate_emg_artifact(n_samples, sfreq, low=20.0, high=200.0):
    """
    Generate muscle artifacts (EMG).
    High frequency (low~high Hz), random bursts.

    Automatically clips the high cutoff to be below Nyquist.
    """
    noise = np.random.randn(n_samples)

    from scipy.signal import butter, filtfilt

    nyq = sfreq / 2.0
    # clip high cutoff to < nyq
    high_clipped = min(high, 0.95 * nyq)

    # if sampling rate too low for desired band, degrade gracefully
    if low >= high_clipped:
        # fallback: just high-pass above (nyq * 0.2) or (low/2), whichever is smaller
        hp = min(max(1.0, nyq * 0.2), 0.95 * nyq)
        b, a = butter(4, hp / nyq, btype='high')
        emg = filtfilt(b, a, noise)
    else:
        b, a = butter(4, [low / nyq, high_clipped / nyq], btype='band')
        emg = filtfilt(b, a, noise)

    # burst envelope
    envelope = np.abs(np.random.randn(max(1, n_samples // 100)))
    envelope = np.repeat(envelope, 100)[:n_samples]

    # smooth envelope (<= 1 Hz low-pass)
    b, a = butter(3, min(1.0, 0.95 * nyq) / nyq, btype='low')
    envelope = filtfilt(b, a, envelope)
    envelope = np.maximum(envelope - 1, 0)

    emg = emg * envelope * 20  # amplitude in "uV-ish units" before 1e-6 scaling
    return emg


def generate_ecg_artifact(n_samples, sfreq, heart_rate=70):
    """
    Generate cardiac artifacts (ECG).

    Periodic pulses at heart rate (~1Hz), small amplitude.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    sfreq : float
        Sampling frequency (Hz)
    heart_rate : float
        Heart rate in BPM (default: 70)

    Returns:
    --------
    ecg : ndarray
        ECG artifact signal
    """
    ecg = np.zeros(n_samples)

    # Heart rate in Hz
    hr_hz = heart_rate / 60

    # R-peak interval (with some variability)
    interval = sfreq / hr_hz
    interval_std = interval * 0.05  # 5% variability (HRV)

    # QRS complex template
    qrs_duration = 0.1  # seconds
    qrs_samples = int(qrs_duration * sfreq)
    qrs_times = np.linspace(-0.05, 0.05, qrs_samples)

    # QRS template: Q (small negative), R (large positive), S (small negative)
    qrs_template = (
        -0.2 * np.exp(-((qrs_times + 0.02) ** 2) / (2 * 0.01 ** 2)) +  # Q
        1.0 * np.exp(-(qrs_times ** 2) / (2 * 0.008 ** 2)) +  # R
        -0.3 * np.exp(-((qrs_times - 0.02) ** 2) / (2 * 0.01 ** 2))  # S
    )

    # Add QRS complexes
    current_pos = int(sfreq)  # Start after 1 second
    while current_pos < n_samples - qrs_samples:
        end_pos = current_pos + qrs_samples
        ecg[current_pos:end_pos] += qrs_template

        # Next beat with variability
        current_pos += int(interval + np.random.randn() * interval_std)

    # Small amplitude (1-2µV)
    ecg = ecg * np.random.uniform(1, 2)

    return ecg


def create_realistic_forward_solution(info, n_sources=10):
    """
    Create more realistic forward solution using actual electrode positions.

    Uses distance-based decay and realistic spatial patterns.

    Parameters:
    -----------
    info : mne.Info
        EEG info with montage
    n_sources : int
        Number of sources

    Returns:
    --------
    fwd_matrix : ndarray
        Forward solution (n_channels, n_sources)
    source_locs : ndarray
        Source locations (n_sources, 3)
    """
    # Get electrode positions
    montage = info.get_montage()
    pos = montage.get_positions()['ch_pos']
    ch_names = info['ch_names']

    # Channel positions (n_channels, 3)
    ch_pos = np.array([pos[ch] for ch in ch_names])

    # Generate random source locations inside the head
    # (simplified: random points within a sphere)
    n_sources = min(n_sources, 20)  # Limit for computational efficiency
    source_locs = np.random.randn(n_sources, 3) * 0.05  # Within ~5cm radius

    # Calculate distances from each source to each electrode
    distances = cdist(source_locs, ch_pos)  # (n_sources, n_channels)

    # Forward model: 1/distance decay (simplified volume conduction)
    fwd_matrix = 1 / (distances.T + 0.01)  # Add small constant to avoid division by zero

    # Add realistic spatial structure
    # Sources in similar regions should project similarly
    for i in range(n_sources):
        # Add some spatial smoothness
        fwd_matrix[:, i] = ndimage.gaussian_filter1d(fwd_matrix[:, i], sigma=1.5)

    # Normalize each source
    fwd_matrix = fwd_matrix / np.max(np.abs(fwd_matrix), axis=0, keepdims=True)

    return fwd_matrix, source_locs


def generate_realistic_sources(n_sources, duration, sfreq):
    """
    Generate realistic brain source signals.

    Combines:
    - 1/f background
    - Burst oscillations
    - Non-stationarity

    Parameters:
    -----------
    n_sources : int
        Number of sources
    duration : float
        Duration (seconds)
    sfreq : float
        Sampling frequency (Hz)

    Returns:
    --------
    sources : ndarray
        Source signals (n_sources, n_samples)
    """
    n_samples = int(duration * sfreq)
    sources = np.zeros((n_sources, n_samples))

    frequency_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    for i in range(n_sources):
        # Base: 1/f noise
        pink = generate_pink_noise(n_samples, alpha=1.5)
        sources[i] = pink * 5

        # Add burst oscillations from 1-2 frequency bands
        n_bands = np.random.randint(1, 3)
        selected_bands = np.random.choice(list(frequency_bands.keys()), n_bands, replace=False)

        for band_name in selected_bands:
            f_low, f_high = frequency_bands[band_name]
            freq = np.random.uniform(f_low, f_high)

            # Alpha has stronger bursts
            burst_rate = 0.4 if band_name == 'alpha' else 0.3

            burst_signal = generate_burst_oscillation(freq, duration, sfreq, burst_rate)
            amp = np.random.uniform(5, 15)
            sources[i] += amp * burst_signal

        # Add slow drift (non-stationarity)
        drift = np.sin(2 * np.pi * 0.05 * np.arange(n_samples) / sfreq)
        sources[i] = sources[i] * (1 + 0.3 * drift)

    return sources


def add_realistic_artifacts(eeg_data, info, sfreq):
    """
    Add realistic artifacts to EEG data with channel-specific patterns.

    Parameters:
    -----------
    eeg_data : ndarray
        EEG data (n_channels, n_samples)
    info : mne.Info
        Channel info
    sfreq : float
        Sampling frequency

    Returns:
    --------
    eeg_with_artifacts : ndarray
        EEG with artifacts
    """
    n_channels, n_samples = eeg_data.shape
    eeg_with_artifacts = eeg_data.copy()

    ch_names = info['ch_names']

    # EOG artifacts (mainly frontal)
    eog = generate_eog_artifact(n_samples, sfreq)
    frontal_weight = np.array([
        1.0 if ch.startswith(('Fp', 'AF', 'F')) else 0.2
        for ch in ch_names
    ])
    eeg_with_artifacts += np.outer(frontal_weight, eog) * 1e-6  # Convert µV to V

    # EMG artifacts (mainly temporal and occipital)
    emg = generate_emg_artifact(n_samples, sfreq)
    temporal_weight = np.array([
        1.0 if any(ch.startswith(prefix) for prefix in ('T', 'TP', 'O', 'PO')) else 0.1
        for ch in ch_names
    ])
    eeg_with_artifacts += np.outer(temporal_weight, emg) * 1e-6

    # ECG artifacts (small, distributed)
    ecg = generate_ecg_artifact(n_samples, sfreq)
    ecg_weight = np.random.uniform(0.3, 0.7, n_channels)  # Variable across channels
    eeg_with_artifacts += np.outer(ecg_weight, ecg) * 1e-6

    # Line noise (50 or 60 Hz)
    line_freq = 60  # Hz (change to 50 for Europe)
    times = np.arange(n_samples) / sfreq
    line_noise = np.sin(2 * np.pi * line_freq * times)
    line_noise_amp = np.random.uniform(0.5, 2, n_channels)  # Variable amplitude
    eeg_with_artifacts += np.outer(line_noise_amp, line_noise) * 1e-6

    return eeg_with_artifacts


def generate_realistic_eeg(output_dir, scenario='realistic', artifact_level='moderate'):
    """
    Generate realistic synthetic EEG data.

    Parameters:
    -----------
    output_dir : Path
        Output directory
    scenario : str
        Scenario name
    artifact_level : str
        'low', 'moderate', 'high'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating realistic EEG: {scenario}, artifacts: {artifact_level}")

    # Parameters
    n_channels = 64
    sfreq = 250
    duration = 60
    n_sources = 10

    # Create info
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    # Generate realistic sources
    print("  Generating sources with 1/f noise and bursts...")
    sources = generate_realistic_sources(n_sources, duration, sfreq)

    # Create realistic forward solution
    print("  Creating forward solution...")
    fwd_matrix, source_locs = create_realistic_forward_solution(info, n_sources)

    # Project to sensors (forward-only baseline)
    eeg_clean = fwd_matrix @ sources  # shape: (n_channels, n_samples)

    # --- Define Ground Truth (target) ---
    # Add small measurement noise to GT (keeps it realistic but still "clean")
    gt_noise_std = 0.5e-6  # adjust if you want; keep smaller than artifacts
    eeg_gt = eeg_clean + (np.random.randn(*eeg_clean.shape) * gt_noise_std)

    # --- Define Observed (input) ---
    artifact_scales = {'low': 0.3, 'moderate': 1.0, 'high': 2.0}
    scale = artifact_scales.get(artifact_level, 1.0)

    print(f"  Adding artifacts to OBS (level: {artifact_level}, scale={scale})...")
    # Add realistic artifacts ONLY to observed signal
    eeg_obs = add_realistic_artifacts(eeg_gt * scale, info, sfreq) / scale

    # Optionally: add a tiny extra sensor noise to OBS (often realistic)
    obs_noise_std = 0.2e-6
    eeg_obs += (np.random.randn(*eeg_obs.shape) * obs_noise_std)

    # --- Create Raw objects ---
    raw_obs = mne.io.RawArray(eeg_obs, info)
    raw_gt  = mne.io.RawArray(eeg_gt, info)
    raw_clean = mne.io.RawArray(eeg_clean, info)

    # Mark bad channels on OBS only (makes sense for "observed data quality")
    n_bad = np.random.randint(3, 8)
    bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
    raw_obs.info['bads'] = bad_channels.tolist()

    print(f"  Bad channels (OBS): {bad_channels.tolist()}")

    # --- Save files (keep your logging style) ---
    # OBS (what you'd feed as input / corrupted observation)
    fif_file = output_dir / f"realistic_eeg_{scenario}_{artifact_level}_raw.fif"
    raw_obs.save(fif_file, overwrite=True)
    print(f"  Saved OBS: {fif_file}")

    # GT (the dense target you want to reconstruct)
    gt_file = output_dir / f"realistic_eeg_{scenario}_{artifact_level}_groundtruth_raw.fif"
    raw_gt.save(gt_file, overwrite=True)
    print(f"  Saved ground truth (GT): {gt_file}")

    # CLEAN (forward-only, no noise/artifacts; for sanity checks)
    clean_file = output_dir / f"realistic_eeg_{scenario}_{artifact_level}_clean_raw.fif"
    raw_clean.save(clean_file, overwrite=True)
    print(f"  Saved clean (forward-only): {clean_file}")

    # Optional extra logging (helps debugging/evaluation)
    print(f"  Shapes: clean={eeg_clean.shape}, gt={eeg_gt.shape}, obs={eeg_obs.shape}")
    print(f"  Noise std: gt_noise={gt_noise_std:.2e}, obs_noise={obs_noise_std:.2e}")

    return raw_obs


def main():
    """Generate realistic datasets."""

    output_dir = Path("/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/synthesis_251222_realistic")

    print("=" * 70)
    print("Realistic EEG Data Generation")
    print("=" * 70)

    # Generate with different artifact levels
    artifact_levels = ['low', 'moderate', 'high']

    for artifact_level in artifact_levels:
        try:
            generate_realistic_eeg(output_dir, scenario='resting',
                                 artifact_level=artifact_level)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
