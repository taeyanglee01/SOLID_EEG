"""
Generate task-specific synthetic EEG data for spatial interpolation testing.

Tasks included:
1. Motor Imagery - Mu/Beta rhythm modulation in sensorimotor cortex
2. Seizure - Spike-wave discharge with spatial propagation
3. P300 Oddball - Event-related potential with parietal maximum
4. Emotion - Frontal alpha asymmetry

References:
- Motor Imagery: https://link.springer.com/article/10.1023/A:1023437823106
- Seizure: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6751861/
- P300: https://pmc.ncbi.nlm.nih.gov/articles/PMC2715154/
- Emotion: https://pmc.ncbi.nlm.nih.gov/articles/PMC6221898/
"""

import numpy as np
import mne
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist


# ============================================================================
# 1. MOTOR IMAGERY TASK
# ============================================================================

def generate_motor_imagery_eeg(info, duration=60, task='left_hand'):
    """
    Generate motor imagery EEG with realistic mu/beta rhythm modulation.

    Motor imagery produces:
    - Mu rhythm (8-12 Hz) desynchronization in contralateral sensorimotor cortex
    - Beta rhythm (13-30 Hz) desynchronization during imagery
    - Beta rebound after imagery
    - Ipsilateral increase in some cases

    Parameters:
    -----------
    info : mne.Info
        EEG info structure
    duration : float
        Duration in seconds
    task : str
        'left_hand', 'right_hand', 'both_hands', 'feet'

    Returns:
    --------
    raw : mne.io.Raw
        Raw EEG data
    events : ndarray
        Event markers (onset times)
    """
    n_channels = len(info['ch_names'])
    sfreq = info['sfreq']
    n_samples = int(duration * sfreq)

    print(f"  Generating motor imagery task: {task}")

    # Base signal: 1/f noise
    from generate_realistic_eeg import generate_pink_noise
    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        eeg_data[ch] = generate_pink_noise(n_samples, alpha=1.5) * 5e-6

    # Add baseline rhythms
    times = np.arange(n_samples) / sfreq

    # Alpha rhythm in occipital (8-13 Hz)
    occipital_chs = [i for i, ch in enumerate(info['ch_names'])
                     if ch.startswith(('O', 'PO', 'P'))]
    alpha_signal = 15e-6 * np.sin(2 * np.pi * 10 * times)
    for ch in occipital_chs:
        eeg_data[ch] += alpha_signal

    # Generate motor imagery trials
    # Paradigm: 3s rest, 4s imagery, 3s rest (10s per trial)
    trial_duration = 10  # seconds
    n_trials = int(duration / trial_duration)

    imagery_onset = 3  # Start imagery at 3s into trial
    imagery_duration = 4  # 4s of imagery

    events = []  # Store event markers

    # Get sensorimotor channel indices
    ch_names = info['ch_names']

    # Central/Sensorimotor channels
    c3_idx = ch_names.index('C3') if 'C3' in ch_names else None
    c4_idx = ch_names.index('C4') if 'C4' in ch_names else None
    cz_idx = ch_names.index('Cz') if 'Cz' in ch_names else None

    # Neighboring channels for smooth spatial distribution
    left_motor = [i for i, ch in enumerate(ch_names)
                  if ch in ['C3', 'C5', 'C1', 'CP3', 'CP5', 'FC3', 'FC5']]
    right_motor = [i for i, ch in enumerate(ch_names)
                   if ch in ['C4', 'C6', 'C2', 'CP4', 'CP6', 'FC4', 'FC6']]
    central = [i for i, ch in enumerate(ch_names)
               if ch in ['Cz', 'CPz', 'FCz']]

    for trial in range(n_trials):
        trial_start = trial * trial_duration
        imagery_start = trial_start + imagery_onset
        imagery_end = imagery_start + imagery_duration

        # Convert to samples
        imagery_start_sample = int(imagery_start * sfreq)
        imagery_end_sample = int(imagery_end * sfreq)

        events.append([imagery_start_sample, 0, 1])  # Event marker

        # Create ERD (Event-Related Desynchronization) envelope
        # Gradual decrease during imagery, then rebound
        trial_samples = int(trial_duration * sfreq)
        trial_times = np.arange(trial_samples) / sfreq

        # Mu rhythm modulation (8-12 Hz)
        mu_freq = 10  # Hz
        mu_base = np.sin(2 * np.pi * mu_freq * trial_times)

        # ERD envelope: decrease during imagery
        erd_envelope = np.ones(trial_samples)
        imagery_samples = np.arange(int(imagery_onset * sfreq),
                                   int((imagery_onset + imagery_duration) * sfreq))

        # Gradual onset and offset
        for i, s in enumerate(imagery_samples):
            if i < sfreq * 0.5:  # 500ms ramp up
                erd_envelope[s] = 1 - 0.6 * (i / (sfreq * 0.5))
            elif i < len(imagery_samples) - sfreq * 0.5:  # Sustained
                erd_envelope[s] = 0.4  # 60% reduction
            else:  # Ramp down
                remaining = len(imagery_samples) - i
                erd_envelope[s] = 0.4 + 0.6 * (1 - remaining / (sfreq * 0.5))

        # Beta rebound after imagery (13-30 Hz)
        beta_freq = 20  # Hz
        beta_base = np.sin(2 * np.pi * beta_freq * trial_times)

        # Beta rebound envelope
        beta_envelope = np.ones(trial_samples)
        rebound_start = int((imagery_onset + imagery_duration) * sfreq)
        rebound_duration = int(1.5 * sfreq)  # 1.5s rebound

        for i in range(min(rebound_duration, trial_samples - rebound_start)):
            s = rebound_start + i
            # Peak at ~500ms after imagery
            peak_time = 0.5 * sfreq
            if i < peak_time:
                beta_envelope[s] = 1 + 0.5 * (i / peak_time)
            else:
                beta_envelope[s] = 1.5 * np.exp(-(i - peak_time) / sfreq)

        # Apply to channels based on task
        trial_start_sample = int(trial_start * sfreq)
        trial_end_sample = min(trial_start_sample + trial_samples, n_samples)
        actual_samples = trial_end_sample - trial_start_sample

        if task == 'left_hand':
            # Contralateral (right hemisphere) desynchronization
            for ch in right_motor:
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    20e-6 * mu_base[:actual_samples] * erd_envelope[:actual_samples]
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    10e-6 * beta_base[:actual_samples] * beta_envelope[:actual_samples]

            # Ipsilateral (left hemisphere) may show slight increase
            for ch in left_motor:
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    15e-6 * mu_base[:actual_samples] * (2 - erd_envelope[:actual_samples])

        elif task == 'right_hand':
            # Contralateral (left hemisphere) desynchronization
            for ch in left_motor:
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    20e-6 * mu_base[:actual_samples] * erd_envelope[:actual_samples]
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    10e-6 * beta_base[:actual_samples] * beta_envelope[:actual_samples]

            # Ipsilateral (right hemisphere)
            for ch in right_motor:
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    15e-6 * mu_base[:actual_samples] * (2 - erd_envelope[:actual_samples])

        elif task == 'both_hands':
            # Bilateral desynchronization
            for ch in left_motor + right_motor:
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    20e-6 * mu_base[:actual_samples] * erd_envelope[:actual_samples]
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    10e-6 * beta_base[:actual_samples] * beta_envelope[:actual_samples]

        elif task == 'feet':
            # Central maximum (Cz)
            for ch in central:
                eeg_data[ch, trial_start_sample:trial_end_sample] += \
                    25e-6 * mu_base[:actual_samples] * erd_envelope[:actual_samples]

    # Add small noise
    eeg_data += np.random.randn(*eeg_data.shape) * 1e-6

    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    events = np.array(events)

    return raw, events


# ============================================================================
# 2. SEIZURE (SPIKE-WAVE DISCHARGE)
# ============================================================================

def generate_seizure_eeg(info, duration=60, seizure_type='absence'):
    """
    Generate seizure EEG with spike-wave discharges and spatial propagation.

    Seizure characteristics:
    - High amplitude (up to 200-500 µV)
    - Spike-wave complexes (sharp spike + slow wave)
    - Spatial propagation from focus to other regions
    - Absence: 3 Hz spike-wave, generalized
    - Focal: starts in one region, spreads

    Parameters:
    -----------
    info : mne.Info
        EEG info structure
    duration : float
        Duration in seconds
    seizure_type : str
        'absence' or 'focal_frontal' or 'focal_temporal'

    Returns:
    --------
    raw : mne.io.Raw
        Raw EEG data
    seizure_events : list
        Seizure onset times
    """
    n_channels = len(info['ch_names'])
    sfreq = info['sfreq']
    n_samples = int(duration * sfreq)

    print(f"  Generating seizure EEG: {seizure_type}")

    # Base signal
    from generate_realistic_eeg import generate_pink_noise
    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        eeg_data[ch] = generate_pink_noise(n_samples, alpha=1.5) * 5e-6

    ch_names = info['ch_names']
    seizure_events = []

    if seizure_type == 'absence':
        # Generalized 3 Hz spike-wave discharge
        # Typically 5-20 seconds duration

        n_seizures = np.random.randint(2, 5)

        for _ in range(n_seizures):
            # Random onset (avoid edges)
            onset_time = np.random.uniform(5, duration - 15)
            seizure_duration = np.random.uniform(5, 15)  # 5-15 seconds

            onset_sample = int(onset_time * sfreq)
            offset_sample = int((onset_time + seizure_duration) * sfreq)

            seizure_events.append([onset_sample, 0, 2])  # Event marker

            # 3 Hz spike-wave complex
            spike_freq = 3  # Hz
            n_spike_waves = int(seizure_duration * spike_freq)

            for i in range(n_spike_waves):
                spike_time = onset_time + i / spike_freq
                spike_sample = int(spike_time * sfreq)

                # Spike-wave complex duration: ~333 ms (1/3 Hz)
                complex_duration = 1 / spike_freq
                complex_samples = int(complex_duration * sfreq)

                if spike_sample + complex_samples > n_samples:
                    break

                # Create spike-wave template
                t = np.linspace(0, complex_duration, complex_samples)

                # Spike: sharp, high amplitude, ~50ms
                spike = -200e-6 * np.exp(-((t - 0.03) ** 2) / (2 * 0.01 ** 2))

                # Wave: slower, ~150ms
                wave = 150e-6 * np.exp(-((t - 0.15) ** 2) / (2 * 0.05 ** 2))

                spike_wave = spike + wave

                # Add to all channels (generalized)
                # With slight amplitude variation and delay
                for ch in range(n_channels):
                    amp_variation = np.random.uniform(0.8, 1.2)
                    delay_samples = np.random.randint(0, int(0.01 * sfreq))  # Up to 10ms delay

                    end_sample = min(spike_sample + delay_samples + complex_samples, n_samples)
                    actual_samples = end_sample - spike_sample - delay_samples

                    eeg_data[ch, spike_sample + delay_samples:end_sample] += \
                        spike_wave[:actual_samples] * amp_variation

    elif seizure_type.startswith('focal'):
        # Focal seizure with propagation
        n_seizures = np.random.randint(1, 3)

        # Determine focus location
        if 'frontal' in seizure_type:
            focus_channels = [i for i, ch in enumerate(ch_names)
                            if ch.startswith(('Fp', 'AF', 'F'))]
        elif 'temporal' in seizure_type:
            focus_channels = [i for i, ch in enumerate(ch_names)
                            if ch.startswith(('T', 'TP', 'FT'))]
        else:
            # Somatosensory (most common for spike initiation)
            focus_channels = [i for i, ch in enumerate(ch_names)
                            if ch.startswith(('C', 'CP'))]

        for _ in range(n_seizures):
            onset_time = np.random.uniform(5, duration - 25)
            seizure_duration = np.random.uniform(10, 30)

            onset_sample = int(onset_time * sfreq)
            seizure_events.append([onset_sample, 0, 3])

            # Propagation pattern: start from focus, spread over ~2-5 seconds
            propagation_duration = np.random.uniform(2, 5)

            # Calculate distances from focus to all channels
            if focus_channels:
                montage = info.get_montage()
                pos = montage.get_positions()['ch_pos']
                ch_pos = np.array([pos[ch] for ch in ch_names])

                focus_center = ch_pos[focus_channels].mean(axis=0)
                distances = np.linalg.norm(ch_pos - focus_center, axis=1)
                max_distance = distances.max()

                # Normalize distances
                normalized_distances = distances / (max_distance + 1e-10)
            else:
                normalized_distances = np.zeros(n_channels)

            # Generate spike train (variable frequency, 1-5 Hz)
            current_time = onset_time
            while current_time < onset_time + seizure_duration:
                spike_time = current_time
                spike_sample = int(spike_time * sfreq)

                # Time since onset (for propagation)
                time_since_onset = spike_time - onset_time

                # Spike template (sharper than absence)
                spike_duration = 0.08  # 80ms
                spike_samples = int(spike_duration * sfreq)
                t = np.linspace(0, spike_duration, spike_samples)

                spike = -300e-6 * np.exp(-((t - 0.02) ** 2) / (2 * 0.005 ** 2))
                spike += 150e-6 * np.exp(-((t - 0.05) ** 2) / (2 * 0.02 ** 2))

                # Add to channels based on propagation
                for ch in range(n_channels):
                    # Propagation delay based on distance
                    propagation_delay = normalized_distances[ch] * propagation_duration

                    if time_since_onset >= propagation_delay:
                        # Amplitude decreases with distance
                        amplitude_factor = 1.0 - 0.5 * normalized_distances[ch]

                        # Add jitter
                        delay_samples = int(propagation_delay * sfreq) + \
                                      np.random.randint(-int(0.01*sfreq), int(0.01*sfreq))

                        end_sample = min(spike_sample + delay_samples + spike_samples, n_samples)
                        actual_samples = end_sample - spike_sample - delay_samples

                        if actual_samples > 0 and spike_sample + delay_samples >= 0:
                            eeg_data[ch, spike_sample + delay_samples:end_sample] += \
                                spike[:actual_samples] * amplitude_factor

                # Next spike (inter-spike interval: 200-1000ms)
                isi = np.random.uniform(0.2, 1.0)
                current_time += isi

    # Add noise
    eeg_data += np.random.randn(*eeg_data.shape) * 2e-6

    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    return raw, np.array(seizure_events) if seizure_events else None


# ============================================================================
# 3. P300 ODDBALL TASK
# ============================================================================

def generate_p300_oddball_eeg(info, duration=300, target_probability=0.2):
    """
    Generate P300 oddball task EEG.

    P300 characteristics:
    - Parietal maximum (Pz, P3, P4)
    - Latency ~300ms post-stimulus
    - Amplitude: 10-20 µV
    - Target stimuli elicit larger P300 than non-targets

    Parameters:
    -----------
    info : mne.Info
        EEG info structure
    duration : float
        Duration in seconds (needs to be longer for multiple trials)
    target_probability : float
        Probability of target stimulus (typically 0.15-0.25)

    Returns:
    --------
    raw : mne.io.Raw
        Raw EEG data
    events : ndarray
        Event markers (sample, 0, event_id)
        event_id: 1=standard, 2=target
    """
    n_channels = len(info['ch_names'])
    sfreq = info['sfreq']
    n_samples = int(duration * sfreq)

    print(f"  Generating P300 oddball task (target probability: {target_probability})")

    # Base signal
    from generate_realistic_eeg import generate_pink_noise
    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        eeg_data[ch] = generate_pink_noise(n_samples, alpha=1.5) * 8e-6

    ch_names = info['ch_names']

    # Parietal channels (P300 maximum)
    parietal_chs = [i for i, ch in enumerate(ch_names)
                   if ch in ['Pz', 'P3', 'P4', 'P1', 'P2', 'POz', 'PO3', 'PO4']]

    # Central channels (moderate P300)
    central_chs = [i for i, ch in enumerate(ch_names)
                  if ch in ['Cz', 'C3', 'C4', 'CPz', 'CP3', 'CP4']]

    # Frontal channels (smaller P300, larger N200)
    frontal_chs = [i for i, ch in enumerate(ch_names)
                  if ch.startswith(('Fp', 'AF', 'F'))]

    # Generate stimulus sequence
    # Inter-stimulus interval: 1-1.5 seconds
    isi_range = (1.0, 1.5)

    events = []
    current_time = 2.0  # Start after 2 seconds

    while current_time < duration - 2:
        # Determine stimulus type
        is_target = np.random.rand() < target_probability

        stimulus_sample = int(current_time * sfreq)
        event_id = 2 if is_target else 1
        events.append([stimulus_sample, 0, event_id])

        # Create ERP template
        # Time windows: 0-800ms post-stimulus
        erp_duration = 0.8  # seconds
        erp_samples = int(erp_duration * sfreq)
        t = np.linspace(0, erp_duration, erp_samples)

        # ERP components:
        # N100: ~100ms, negative
        n100 = -8e-6 * np.exp(-((t - 0.1) ** 2) / (2 * 0.02 ** 2))

        # P200: ~200ms, positive
        p200 = 6e-6 * np.exp(-((t - 0.2) ** 2) / (2 * 0.03 ** 2))

        # N200: ~250ms, negative (larger for targets)
        n200_latency = 0.25
        n200_amp = -10e-6 if is_target else -5e-6
        n200 = n200_amp * np.exp(-((t - n200_latency) ** 2) / (2 * 0.03 ** 2))

        # P300: ~300-350ms, positive (much larger for targets)
        p300_latency = np.random.uniform(0.28, 0.35) if is_target else 0.32
        p300_amp = np.random.uniform(15e-6, 25e-6) if is_target else np.random.uniform(5e-6, 8e-6)
        p300 = p300_amp * np.exp(-((t - p300_latency) ** 2) / (2 * 0.06 ** 2))

        # Combine components
        erp = n100 + p200 + n200 + p300

        # Add to channels with spatial distribution
        end_sample = min(stimulus_sample + erp_samples, n_samples)
        actual_samples = end_sample - stimulus_sample

        # Parietal maximum for P300
        for ch in parietal_chs:
            eeg_data[ch, stimulus_sample:end_sample] += erp[:actual_samples] * 1.0

        # Central: moderate
        for ch in central_chs:
            eeg_data[ch, stimulus_sample:end_sample] += erp[:actual_samples] * 0.7

        # Frontal: smaller P300, but larger N200
        frontal_erp = n100 + p200 + n200 * 1.5 + p300 * 0.3
        for ch in frontal_chs:
            eeg_data[ch, stimulus_sample:end_sample] += frontal_erp[:actual_samples]

        # Next stimulus
        isi = np.random.uniform(*isi_range)
        current_time += isi

    # Add noise
    eeg_data += np.random.randn(*eeg_data.shape) * 3e-6

    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    events = np.array(events)

    print(f"    Generated {len(events)} stimuli ({np.sum(events[:, 2] == 2)} targets)")

    return raw, events


# ============================================================================
# 4. EMOTION (FRONTAL ALPHA ASYMMETRY)
# ============================================================================

def generate_emotion_eeg(info, duration=120, emotion='positive'):
    """
    Generate emotion-related EEG with frontal alpha asymmetry.

    Emotion characteristics:
    - Frontal alpha asymmetry (F3 vs F4)
    - Positive emotion: Greater left frontal activity (lower alpha in F3)
    - Negative emotion: Greater right frontal activity (lower alpha in F4)
    - Theta in frontal midline for emotional arousal

    Parameters:
    -----------
    info : mne.Info
        EEG info structure
    duration : float
        Duration in seconds
    emotion : str
        'positive', 'negative', 'neutral', 'arousal'

    Returns:
    --------
    raw : mne.io.Raw
        Raw EEG data
    """
    n_channels = len(info['ch_names'])
    sfreq = info['sfreq']
    n_samples = int(duration * sfreq)

    print(f"  Generating emotion EEG: {emotion}")

    # Base signal
    from generate_realistic_eeg import generate_pink_noise
    eeg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        eeg_data[ch] = generate_pink_noise(n_samples, alpha=1.5) * 5e-6

    ch_names = info['ch_names']
    times = np.arange(n_samples) / sfreq

    # Alpha rhythm (8-13 Hz) - key for frontal asymmetry
    alpha_freq = 10  # Hz

    # Get channel indices
    f3_idx = ch_names.index('F3') if 'F3' in ch_names else None
    f4_idx = ch_names.index('F4') if 'F4' in ch_names else None
    fp1_idx = ch_names.index('Fp1') if 'Fp1' in ch_names else None
    fp2_idx = ch_names.index('Fp2') if 'Fp2' in ch_names else None

    # Left frontal
    left_frontal = [i for i, ch in enumerate(ch_names)
                   if ch in ['F3', 'F1', 'F5', 'AF3', 'FC3', 'FC5']]

    # Right frontal
    right_frontal = [i for i, ch in enumerate(ch_names)
                    if ch in ['F4', 'F2', 'F6', 'AF4', 'FC4', 'FC6']]

    # Frontal midline
    frontal_midline = [i for i, ch in enumerate(ch_names)
                      if ch in ['Fz', 'Fpz', 'AFz', 'FCz']]

    # Generate alpha rhythm with asymmetry
    alpha_base = np.sin(2 * np.pi * alpha_freq * times)

    # Create burst envelope (alpha comes and goes)
    from generate_realistic_eeg import generate_burst_oscillation
    alpha_envelope = np.abs(generate_burst_oscillation(0.1, duration, sfreq, burst_rate=0.6))
    alpha_envelope = alpha_envelope / (np.max(alpha_envelope) + 1e-10)

    if emotion == 'positive':
        # Positive: Left frontal activation = LOWER alpha in left
        # (Alpha power inversely related to activation)
        left_alpha_amp = 12e-6  # Lower alpha
        right_alpha_amp = 20e-6  # Higher alpha

    elif emotion == 'negative':
        # Negative: Right frontal activation = LOWER alpha in right
        left_alpha_amp = 20e-6  # Higher alpha
        right_alpha_amp = 12e-6  # Lower alpha

    else:  # neutral
        left_alpha_amp = 16e-6
        right_alpha_amp = 16e-6

    # Apply to left frontal
    for ch in left_frontal:
        eeg_data[ch] += left_alpha_amp * alpha_base * alpha_envelope

    # Apply to right frontal
    for ch in right_frontal:
        eeg_data[ch] += right_alpha_amp * alpha_base * alpha_envelope

    # Frontal midline theta (Fm theta) for arousal/emotion
    theta_freq = 6  # Hz
    theta_base = np.sin(2 * np.pi * theta_freq * times)

    if emotion == 'arousal' or emotion in ['positive', 'negative']:
        # Increased frontal theta during emotional processing
        theta_amp = 10e-6
        for ch in frontal_midline:
            eeg_data[ch] += theta_amp * theta_base

    # Add posterior alpha (occipital)
    occipital_chs = [i for i, ch in enumerate(ch_names)
                    if ch.startswith(('O', 'PO'))]
    for ch in occipital_chs:
        eeg_data[ch] += 18e-6 * alpha_base * alpha_envelope

    # Add slow drift (emotional state changes slowly)
    drift_freq = 0.02  # Very slow
    drift = np.sin(2 * np.pi * drift_freq * times)
    eeg_data = eeg_data * (1 + 0.2 * drift[:, np.newaxis].T)

    # Add noise
    eeg_data += np.random.randn(*eeg_data.shape) * 2e-6

    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    return raw


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate all task-specific EEG datasets."""

    output_dir = Path("/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/synthesis_251223_task")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Task-Specific EEG Data Generation")
    print("=" * 70)

    # Common parameters
    n_channels = 64
    sfreq = 250

    # Create montage and info
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    # -------------------------------------------------------------------------
    # 1. Motor Imagery Tasks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. MOTOR IMAGERY TASKS")
    print("=" * 70)

    for task in ['left_hand', 'right_hand', 'both_hands', 'feet']:
        try:
            raw, events = generate_motor_imagery_eeg(info, duration=60, task=task)

            # Mark bad channels
            n_bad = np.random.randint(3, 6)
            bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
            raw.info['bads'] = bad_channels.tolist()

            # Save
            filename = output_dir / f"motor_imagery_{task}_raw.fif"
            raw.save(filename, overwrite=True)
            print(f"  Saved: {filename}")

            # Save ground truth
            raw_gt = raw.copy()
            raw_gt.info['bads'] = []
            gt_filename = output_dir / f"motor_imagery_{task}_groundtruth_raw.fif"
            raw_gt.save(gt_filename, overwrite=True)

            # Save events
            events_filename = output_dir / f"motor_imagery_{task}_events.txt"
            np.savetxt(events_filename, events, fmt='%d')

        except Exception as e:
            print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # 2. Seizure
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. SEIZURE TASKS")
    print("=" * 70)

    for seizure_type in ['absence', 'focal_frontal', 'focal_temporal']:
        try:
            raw, events = generate_seizure_eeg(info, duration=60, seizure_type=seizure_type)

            n_bad = np.random.randint(3, 6)
            bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
            raw.info['bads'] = bad_channels.tolist()

            filename = output_dir / f"seizure_{seizure_type}_raw.fif"
            raw.save(filename, overwrite=True)
            print(f"  Saved: {filename}")

            raw_gt = raw.copy()
            raw_gt.info['bads'] = []
            gt_filename = output_dir / f"seizure_{seizure_type}_groundtruth_raw.fif"
            raw_gt.save(gt_filename, overwrite=True)

            if events is not None:
                events_filename = output_dir / f"seizure_{seizure_type}_events.txt"
                np.savetxt(events_filename, events, fmt='%d')

        except Exception as e:
            print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # 3. P300 Oddball
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. P300 ODDBALL TASK")
    print("=" * 70)

    try:
        raw, events = generate_p300_oddball_eeg(info, duration=300, target_probability=0.2)

        n_bad = np.random.randint(3, 6)
        bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
        raw.info['bads'] = bad_channels.tolist()

        filename = output_dir / "p300_oddball_raw.fif"
        raw.save(filename, overwrite=True)
        print(f"  Saved: {filename}")

        raw_gt = raw.copy()
        raw_gt.info['bads'] = []
        gt_filename = output_dir / "p300_oddball_groundtruth_raw.fif"
        raw_gt.save(gt_filename, overwrite=True)

        events_filename = output_dir / "p300_oddball_events.txt"
        np.savetxt(events_filename, events, fmt='%d')

    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # 4. Emotion
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. EMOTION TASKS")
    print("=" * 70)

    for emotion in ['positive', 'negative', 'neutral']:
        try:
            raw = generate_emotion_eeg(info, duration=120, emotion=emotion)

            n_bad = np.random.randint(3, 6)
            bad_channels = np.random.choice(ch_names, size=n_bad, replace=False)
            raw.info['bads'] = bad_channels.tolist()

            filename = output_dir / f"emotion_{emotion}_raw.fif"
            raw.save(filename, overwrite=True)
            print(f"  Saved: {filename}")

            raw_gt = raw.copy()
            raw_gt.info['bads'] = []
            gt_filename = output_dir / f"emotion_{emotion}_groundtruth_raw.fif"
            raw_gt.save(gt_filename, overwrite=True)

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Complete! All task-specific EEG data generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
