#!/bin/bash
#SBATCH -A m4727_g
#SBATCH -C gpu #&hbm80g
#SBATCH -q shared #preempt #regular #shared #regular, shared,  #! 30 mins is enough so debug
#SBATCH --job-name=SOPh1227
#SBATCH --output=/pscratch/sd/t/tylee/slurm_outputs/solid/251227_SOLID_Physio_sample_test-%A_%a.out
#SBATCH --error=/pscratch/sd/t/tylee/slurm_outputs/solid/251227_SOLID_Physio_sample_test-%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00

# ==============================================================================
# Multi-Subject EEG Data Generation Script
# ==============================================================================
# This script generates synthetic EEG data for multiple subjects
# using the generate_task_specific_eeg_multisubject.py script.
# ==============================================================================

# Configuration
N_SUBJECTS=50
OUTPUT_DIR="/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_data"
N_CHANNELS=64
SFREQ=200

# Task durations (in seconds)
MOTOR_DURATION=120
SEIZURE_DURATION=120
P300_DURATION=600
EMOTION_DURATION=240

# Task selection (all, motor, seizure, p300, emotion, or comma-separated)
TASKS="all"

# ==============================================================================
# Setup
# ==============================================================================

# Print job information

# Print configuration
echo "Configuration:"
echo "  Number of subjects: $N_SUBJECTS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of channels: $N_CHANNELS"
echo "  Sampling frequency: $SFREQ Hz"
echo "  Tasks: $TASKS"
echo "  Motor Imagery duration: $MOTOR_DURATION s"
echo "  Seizure duration: $SEIZURE_DURATION s"
echo "  P300 duration: $P300_DURATION s"
echo "  Emotion duration: $EMOTION_DURATION s"
echo ""

# ==============================================================================
# Environment Setup
# ==============================================================================

# Load required modules (adjust based on your cluster)
# module load python/3.9
# module load conda

# Activate conda environment
# Replace 'your_eeg_env' with your actual environment name
module load conda
conda activate fingerflex

cd /pscratch/sd/t/tylee/SOLID_EEG/synthetic


# ==============================================================================
# Run Data Generation
# ==============================================================================

echo "Starting data generation..."
echo ""

# Run the Python script
python generate_task_specific_eeg_multisubject.py \
    --n_subjects $N_SUBJECTS \
    --output_dir $OUTPUT_DIR \
    --n_channels $N_CHANNELS \
    --sfreq $SFREQ \
    --tasks $TASKS \
    --motor_duration $MOTOR_DURATION \
    --seizure_duration $SEIZURE_DURATION \
    --p300_duration $P300_DURATION \
    --emotion_duration $EMOTION_DURATION


# ==============================================================================
# Optional: Create metadata file
# ==============================================================================

METADATA_FILE="$OUTPUT_DIR/dataset_metadata.txt"

cat > $METADATA_FILE << EOF
Multi-Subject Task-Specific EEG Dataset
========================================

Generation Date: $(date)
Job ID: $SLURM_JOB_ID
Node: $SLURM_NODELIST

Dataset Configuration:
----------------------
Number of subjects: $N_SUBJECTS
Number of channels: $N_CHANNELS
Sampling frequency: $SFREQ Hz
Tasks generated: $TASKS

Task Durations:
--------------
Motor Imagery: $MOTOR_DURATION seconds
Seizure: $SEIZURE_DURATION seconds
P300 Oddball: $P300_DURATION seconds
Emotion: $EMOTION_DURATION seconds

Directory Structure:
-------------------
Each subject has a directory named 'sub-XXX' containing:
- Motor imagery tasks (4 conditions: left_hand, right_hand, both_hands, feet)
- Seizure recordings (3 types: absence, focal_frontal, focal_temporal)
- P300 oddball task
- Emotion tasks (3 conditions: positive, negative, neutral)

File Types:
----------
For each task:
- *_raw.fif: Raw EEG data with bad channels marked
- *_groundtruth_raw.fif: Ground truth data without bad channels
- *_events.txt: Event markers (where applicable)

Subject Variations:
------------------
Each subject has unique variations in:
- Amplitude scaling: 0.8 to 1.2 (±20%)
- Frequency shift: 0.95 to 1.05 (±5%)

Generation Script:
-----------------
generate_task_specific_eeg_multisubject.py

Notes:
------
This dataset is intended for testing spatial interpolation methods
and other EEG analysis pipelines.

EOF

echo "Metadata file created: $METADATA_FILE"
echo ""

# ==============================================================================
# Cleanup and finish
# ==============================================================================

echo "Job completed at: $(date)"
echo ""

# Exit successfully
exit 0
