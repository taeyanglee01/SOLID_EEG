# Task-Specific EEG Data for Spatial Interpolation Testing

This document describes task-specific synthetic EEG datasets designed to test spatial interpolation algorithms under realistic neuroscientific scenarios.

## Why Task-Specific Data?

Task-specific EEG has **known spatial-temporal patterns** that make interpolation testing more meaningful:
- ✅ **Realistic spatial structure**: Not random, but neurophysiologically plausible
- ✅ **Quantifiable metrics**: Can measure if specific patterns (e.g., lateralization, topography) are preserved
- ✅ **Clinical relevance**: Tests interpolation in scenarios encountered in real research/clinical settings
- ✅ **Frequency-specific**: Tests interpolation across different frequency bands

---

## 1. Motor Imagery 🤚

### Neuroscience Background

**Key Papers:**
- [Mu and Beta Rhythm Topographies During Motor Imagery](https://link.springer.com/article/10.1023/A:1023437823106)
- [Motor Imagery Classification Using Mu and Beta Rhythms](https://pmc.ncbi.nlm.nih.gov/articles/PMC5066028/)

**Neural Mechanisms:**
- **Mu rhythm (8-12 Hz)**: Desynchronization (ERD) in contralateral sensorimotor cortex during motor imagery
- **Beta rhythm (13-30 Hz)**: Desynchronization during imagery, rebound (~20% increase) after
- **Spatial specificity**:
  - Left hand → Right motor cortex (C4, C6, CP4)
  - Right hand → Left motor cortex (C3, C5, CP3)
  - Feet → Midline (Cz, CPz)

### Generated Tasks

1. **Left hand imagery**: Mu/beta ERD in right hemisphere (C4 focus)
2. **Right hand imagery**: Mu/beta ERD in left hemisphere (C3 focus)
3. **Both hands imagery**: Bilateral ERD
4. **Feet imagery**: Midline ERD (Cz focus)

### Interpolation Testing Value

| Test | What It Evaluates |
|------|-------------------|
| **C3 bad, right hand task** | Can interpolation recover contralateral desynchronization? |
| **C4 bad, left hand task** | Same as above |
| **C3 + C4 bad** | Can bilateral pattern be recovered? |
| **Frequency-specific** | Does interpolation preserve mu (8-12 Hz) vs beta (13-30 Hz) differently? |

**Critical Test:**
```python
# If C3 is bad during right hand imagery:
# - C3 should show STRONG mu/beta ERD (contralateral)
# - C4 should show WEAK/absent ERD (ipsilateral)
# Can interpolation distinguish this?
```

**Expected Challenge:**
- Interpolation must NOT just average neighboring channels
- Must preserve lateralization (C3 ≠ C4 during unilateral imagery)

---

## 2. Seizure (Spike-Wave Discharge) ⚡

### Neuroscience Background

**Key Papers:**
- [Distinct Topographical Patterns of Spike-Wave Discharge](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6751861/)
- [EEG Abnormal Waveforms](https://www.ncbi.nlm.nih.gov/books/NBK557655/)

**Neural Mechanisms:**
- **Absence seizure**: Generalized 3 Hz spike-wave complex, simultaneous across all channels
- **Focal seizure**:
  - Initiates in focal region (e.g., somatosensory cortex)
  - Propagates to adjacent areas over 2-5 seconds
  - Spike frequency: 1-5 Hz (variable)
- **High amplitude**: 200-500 µV (vs normal EEG ~10-50 µV)

### Generated Tasks

1. **Absence seizure**: 3 Hz generalized spike-wave
2. **Focal frontal seizure**: Starts in frontal region, spreads posteriorly
3. **Focal temporal seizure**: Starts in temporal region, spreads

### Interpolation Testing Value

| Test | What It Evaluates |
|------|-------------------|
| **Bad channel in seizure focus** | Can interpolation recover ictal activity from neighbors? |
| **Bad channel in propagation path** | Does interpolation track spatial-temporal spreading? |
| **High amplitude handling** | Can interpolation handle extreme amplitudes without artifacts? |
| **Sharp transients** | Are spike morphologies preserved? |

**Critical Test:**
```python
# Focal frontal seizure with F3 bad:
# - Seizure starts in frontal region (including F3)
# - Spreads to central/posterior over 3 seconds
# Questions:
# 1. Can interpolation recover initial spike at F3?
# 2. Does it correctly show propagation timing?
```

**Expected Challenge:**
- Very high amplitude may confuse interpolation algorithms
- Spike-wave morphology is sharp → requires preserving high frequencies
- Propagation creates non-stationary spatial patterns

**Clinical Relevance:**
- Critical for seizure onset zone localization
- Bad channels in seizure focus = lost diagnostic information

---

## 3. P300 Oddball Task 🎯

### Neuroscience Background

**Key Papers:**
- [Updating P300: An Integrative Theory of P3a and P3b](https://pmc.ncbi.nlm.nih.gov/articles/PMC2715154/)
- [Visual Selective Attention P300 Source](https://link.springer.com/article/10.1007/s10548-022-00916-x)

**Neural Mechanisms:**
- **P3b component**: ~300ms post-stimulus, **parietal maximum** (Pz, P3, P4)
- **Amplitude**: 10-25 µV for targets, 5-8 µV for non-targets
- **Topography**: Highest at Pz, decreases frontally and laterally
- **N200**: ~250ms, frontal maximum, larger for targets

### Generated Tasks

- **Oddball paradigm**: 20% targets, 80% standards
- **~200 trials** over 5 minutes
- **Event markers**: Stimulus onset times saved

### Interpolation Testing Value

| Test | What It Evaluates |
|------|-------------------|
| **Pz bad** | Can interpolation recover maximum P300? |
| **P3 or P4 bad** | Asymmetry in parietal distribution preserved? |
| **Frontal channels bad** | Does interpolation preserve frontal-parietal gradient? |
| **ERP morphology** | Are peak latency and amplitude preserved? |

**Critical Test:**
```python
# With Pz bad (parietal maximum):
# Expected P300 at Pz: ~20 µV
# Neighbors (P3, P4): ~15 µV
# Neighbors (CPz, POz): ~12 µV

# Can interpolation:
# 1. Estimate Pz amplitude correctly (not just average neighbors)?
# 2. Preserve peak latency (~300ms)?
# 3. Maintain topographic gradient?
```

**Analysis Approach:**
```python
# Average ERP across trials
epochs = mne.Epochs(raw, events, event_id={'target': 2},
                    tmin=-0.2, tmax=0.8)
evoked = epochs.average()

# Compare Pz interpolated vs ground truth
evoked_interp = epochs_interpolated.average()
evoked_gt = epochs_groundtruth.average()

# Metrics:
# - Peak amplitude difference
# - Peak latency difference
# - Topographic similarity (correlation)
```

**Expected Challenge:**
- ERPs are time-locked and averaged across trials
- Small errors can compound across trials
- Topography must be precise for source localization

---

## 4. Emotion (Frontal Alpha Asymmetry) 😊😢

### Neuroscience Background

**Key Papers:**
- [Frontal EEG Asymmetry and Middle Line Power Difference](https://pmc.ncbi.nlm.nih.gov/articles/PMC6221898/)
- [Identifying relevant asymmetry features of EEG](https://pmc.ncbi.nlm.nih.gov/articles/PMC10469865/)

**Neural Mechanisms:**
- **Frontal alpha asymmetry**: Key marker for emotional valence
  - **Left frontal activation** (lower alpha at F3): Positive emotion, approach
  - **Right frontal activation** (lower alpha at F4): Negative emotion, withdrawal
- **Asymmetry index**: `AI = log(F4_alpha) - log(F3_alpha)`
- **Frontal-midline theta**: Increased during emotional arousal

### Generated Tasks

1. **Positive emotion**: F3 alpha < F4 alpha (left activation)
2. **Negative emotion**: F4 alpha < F3 alpha (right activation)
3. **Neutral**: F3 alpha ≈ F4 alpha

### Interpolation Testing Value

| Test | What It Evaluates |
|------|-------------------|
| **F3 bad** | Can asymmetry index be computed correctly? |
| **F4 bad** | Same as above |
| **F3 AND F4 bad** | Complete loss of asymmetry information? |
| **Homologous channels** | Are left-right relationships preserved? |

**Critical Test:**
```python
# Positive emotion: F3 alpha = 12 µV, F4 alpha = 20 µV
# Asymmetry Index (AI) = log(20) - log(12) = 0.51

# If F3 is bad and interpolated:
# F3_interp = weighted_avg(F1, F5, AF3, FC3, ...)

# Questions:
# 1. Is F3_interp closer to F3_true or F4?
# 2. Is AI preserved? (critical for emotion classification)
```

**Analysis Approach:**
```python
# Extract alpha power
from mne.time_frequency import psd_welch

# Alpha band: 8-13 Hz
psd, freqs = psd_welch(raw, fmin=8, fmax=13, picks='eeg')
alpha_power = psd.mean(axis=1)  # Average across frequencies

# Compute asymmetry
f3_idx = raw.ch_names.index('F3')
f4_idx = raw.ch_names.index('F4')

ai_interp = np.log(alpha_power[f4_idx]) - np.log(alpha_power[f3_idx])
ai_gt = np.log(alpha_power_gt[f4_idx]) - np.log(alpha_power_gt[f3_idx])

# Critical: Does interpolation preserve asymmetry direction?
```

**Expected Challenge:**
- Alpha asymmetry is RELATIVE (F3 vs F4), not absolute
- Interpolation must preserve left-right relationships
- Small errors in power estimates → large errors in AI

**Clinical Relevance:**
- Depression screening (abnormal frontal asymmetry)
- Emotion recognition for BCI
- Treatment response prediction

---

## Summary: Which Task for Which Test?

| Interpolation Aspect | Best Task |
|----------------------|-----------|
| **Lateralization preservation** | Motor Imagery |
| **High amplitude handling** | Seizure |
| **Sharp transients** | Seizure |
| **Topographic accuracy** | P300 |
| **Time-locked patterns** | P300 |
| **Frequency-specific** | Emotion (alpha) |
| **Symmetric channel relationships** | Emotion (F3/F4) |
| **Spatial-temporal dynamics** | Seizure (propagation) |

---

## Usage Example

### Basic Testing

```python
import mne
import numpy as np
from pathlib import Path

# Load motor imagery data
data_dir = Path("task_specific_data")

# Right hand motor imagery
raw = mne.io.read_raw_fif(data_dir / "motor_imagery_right_hand_raw.fif", preload=True)
raw_gt = mne.io.read_raw_fif(data_dir / "motor_imagery_right_hand_groundtruth_raw.fif", preload=True)

bad_channels = raw.info['bads']
print(f"Bad channels: {bad_channels}")

# Your interpolation
raw_interp = your_interpolation_method(raw)

# Extract mu rhythm (8-12 Hz) at C3 (should show ERD)
from scipy import signal as sp_signal

# Bandpass filter
raw_mu = raw_interp.copy().filter(8, 12)
raw_mu_gt = raw_gt.copy().filter(8, 12)

# Get C3 data
c3_data = raw_mu.get_data(picks=['C3'])[0]
c3_gt = raw_mu_gt.get_data(picks=['C3'])[0]

# Check if ERD pattern is preserved
correlation = np.corrcoef(c3_data, c3_gt)[0, 1]
print(f"C3 mu rhythm correlation: {correlation:.3f}")
```

### Advanced: Task-Specific Metrics

```python
# Motor Imagery: Lateralization Index
def compute_lateralization_index(raw, task='right_hand'):
    """
    For right hand: Should be negative (C3 < C4)
    For left hand: Should be positive (C3 > C4)
    """
    mu_power_c3 = compute_band_power(raw, 'C3', 8, 12)
    mu_power_c4 = compute_band_power(raw, 'C4', 8, 12)

    li = (mu_power_c3 - mu_power_c4) / (mu_power_c3 + mu_power_c4)
    return li

# P300: Peak Topography Similarity
def compute_topo_similarity(evoked_interp, evoked_gt, tmin=0.25, tmax=0.4):
    """
    Spatial correlation of P300 topography
    """
    # Extract P300 time window
    p300_interp = evoked_interp.copy().crop(tmin, tmax).data.mean(axis=1)
    p300_gt = evoked_gt.copy().crop(tmin, tmax).data.mean(axis=1)

    # Spatial correlation
    corr = np.corrcoef(p300_interp, p300_gt)[0, 1]
    return corr

# Emotion: Asymmetry Index Preservation
def compute_asymmetry_index(raw):
    """
    AI = log(F4_alpha) - log(F3_alpha)
    """
    f3_alpha = compute_band_power(raw, 'F3', 8, 13)
    f4_alpha = compute_band_power(raw, 'F4', 8, 13)

    ai = np.log(f4_alpha) - np.log(f3_alpha)
    return ai
```

---

## Recommended Testing Protocol

### Level 1: Single-channel bad (easiest)
- One bad channel per task
- Not in critical location (e.g., NOT Pz for P300)

### Level 2: Critical-channel bad (harder)
- Pz for P300
- C3 for right-hand motor imagery
- F3 for positive emotion

### Level 3: Multiple bad channels (hardest)
- 3-5 bad channels
- Some in critical locations
- Test spatial pattern recovery

### Level 4: Worst-case scenarios
- All parietal channels bad for P300
- Both C3 and C4 bad for motor imagery
- Both F3 and F4 bad for emotion

---

## Expected Performance Benchmarks

Based on MNE-Python's spherical spline interpolation:

| Task | Metric | Good Performance | Acceptable | Poor |
|------|--------|------------------|------------|------|
| Motor Imagery | Lateralization Index preservation | > 0.90 | 0.75-0.90 | < 0.75 |
| Seizure | Spike morphology correlation | > 0.85 | 0.70-0.85 | < 0.70 |
| P300 | Topography correlation | > 0.92 | 0.80-0.92 | < 0.80 |
| Emotion | Asymmetry Index error | < 0.1 | 0.1-0.2 | > 0.2 |

---

## Files Generated

For each task, three files are created:

1. `{task}_raw.fif`: Data with bad channels marked
2. `{task}_groundtruth_raw.fif`: Complete data (no bad channels)
3. `{task}_events.txt`: Event markers (for P300, motor imagery, seizure)

**Total datasets:** 13 task-specific datasets
- 4 motor imagery (left, right, both, feet)
- 3 seizure (absence, focal frontal, focal temporal)
- 1 P300 oddball
- 3 emotion (positive, negative, neutral)
- 2 realistic (from previous script)

---

## References

### Motor Imagery
- [Mu and Beta Rhythm Topographies](https://link.springer.com/article/10.1023/A:1023437823106)
- [Motor Imagery Classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC5066028/)

### Seizure
- [Spike-Wave Discharge Patterns](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6751861/)
- [EEG Abnormal Waveforms](https://www.ncbi.nlm.nih.gov/books/NBK557655/)

### P300
- [Updating P300 Theory](https://pmc.ncbi.nlm.nih.gov/articles/PMC2715154/)
- [P300 Source Localization](https://link.springer.com/article/10.1007/s10548-022-00916-x)

### Emotion
- [Frontal Alpha Asymmetry](https://pmc.ncbi.nlm.nih.gov/articles/PMC6221898/)
- [EEG Emotion Recognition](https://pmc.ncbi.nlm.nih.gov/articles/PMC10469865/)
