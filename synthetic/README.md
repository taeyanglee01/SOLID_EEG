# SOLID_EEG
Project for apply SOLID model to EEG dataset

## Synthetic EEG Data Generation

### Overview

The `generate_synthetic_eeg.py` script creates realistic synthetic EEG datasets with:
- Known ground truth (before interpolation)
- Marked bad channels for interpolation testing
- Multiple scenarios (different SNR levels, event-related)
- Spatial pattern validation datasets

### Requirements

```bash
pip install mne numpy scipy
```

### Usage

Run the data generation script:

```bash
python generate_synthetic_eeg.py
```

This will create the `synthetic_data/` directory with multiple datasets.

### Generated Datasets

#### 1. Scenario-based Datasets

Four different scenarios are generated, each saved in multiple formats:

- **default**: Standard EEG with moderate noise (SNR ~10)
- **high_snr**: Clean EEG with minimal noise (SNR ~100)
- **low_snr**: Noisy EEG (SNR ~2)
- **event_related**: EEG with event-related potentials (ERPs)

**Files for each scenario:**
- `synthetic_eeg_{scenario}_raw.fif`: Data with bad channels marked
- `synthetic_eeg_{scenario}.edf`: Same data in EDF format
- `synthetic_eeg_{scenario}_groundtruth_raw.fif`: Clean data without bad channels
- `synthetic_eeg_{scenario}_metadata.json`: Metadata including bad channel list

#### 2. Spatial Pattern Validation Dataset

Located in `synthetic_data/validation/`:

- **spatial_pattern_groundtruth_raw.fif**: Complete dataset with known spatial patterns
- **spatial_pattern_random_raw.fif**: 5 random bad channels
- **spatial_pattern_cluster_raw.fif**: Clustered bad channels
- **spatial_pattern_frontal_raw.fif**: Bad channels in frontal region
- **spatial_pattern_temporal_raw.fif**: Bad channels in temporal region

These datasets have known spatial gradients (left-right, front-back, radial) that can be used to validate interpolation accuracy.

### Data Characteristics

- **Number of channels**: 64 (standard 10-20 system)
- **Sampling rate**: 250 Hz
- **Duration**: 60 seconds (scenario datasets), 10 seconds (validation dataset)
- **Bad channels**: 3-7 randomly selected per scenario

### Signal Components

The synthetic EEG contains:
1. **Multiple frequency bands**: Delta (1 Hz), Theta (4 Hz), Alpha (8 Hz), Beta (13 Hz), Gamma (30 Hz)
2. **Spatial structure**: Realistic spatial mixing of sources
3. **Noise**: Additive Gaussian noise with scenario-specific levels
4. **ERPs** (event_related scenario): P300-like components

### Testing Interpolation Models

**Basic Workflow:**

```python
import mne

# Load data with bad channels
raw = mne.io.read_raw_fif('synthetic_data/synthetic_eeg_default_raw.fif', preload=True)

# Your interpolation here
raw.interpolate_bads()

# Load ground truth for comparison
raw_gt = mne.io.read_raw_fif('synthetic_data/synthetic_eeg_default_groundtruth_raw.fif', preload=True)

# Compare
data_interpolated = raw.get_data()
data_groundtruth = raw_gt.get_data()

# Calculate metrics (e.g., RMSE, correlation)
```

**Validation with Known Spatial Patterns:**

```python
import mne
import numpy as np

# Load ground truth
raw_gt = mne.io.read_raw_fif('synthetic_data/validation/spatial_pattern_groundtruth_raw.fif')

# Load test data with bad channels
raw_test = mne.io.read_raw_fif('synthetic_data/validation/spatial_pattern_random_raw.fif', preload=True)

bad_channels = raw_test.info['bads']

# Extract ground truth for bad channels
gt_data = raw_gt.get_data(picks=bad_channels)

# Apply your interpolation
raw_interpolated = raw_test.copy()
raw_interpolated.interpolate_bads()  # or your custom method

# Get interpolated data for previously bad channels
interp_data = raw_interpolated.get_data(picks=bad_channels)

# Calculate accuracy
rmse = np.sqrt(np.mean((gt_data - interp_data)**2))
correlation = np.corrcoef(gt_data.flatten(), interp_data.flatten())[0, 1]

print(f"RMSE: {rmse}")
print(f"Correlation: {correlation}")
```
