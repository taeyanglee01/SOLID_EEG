#!/bin/bash
#SBATCH --job-name=gen_multisubject_eeg
#SBATCH --output=logs/multisubject_eeg_%j.out
#SBATCH --error=logs/multisubject_eeg_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --account=m4239
#SBATCH --constraint=cpu
#SBATCH --qos=regular

# ==============================================================================
# Multi-Subject EEG Data Generation Script
# ==============================================================================
# This script generates synthetic EEG data for multiple subjects
# using the generate_task_specific_eeg_multisubject.py script.
# ==============================================================================

# Configuration
N_SUBJECTS=100
OUTPUT_DIR="/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_data_$(date +%y%m%d)"
N_CHANNELS=64
SFREQ=250

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

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "===================================================================="
echo "Multi-Subject EEG Data Generation"
echo "===================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "===================================================================="
echo ""

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
source activate base
# conda activate your_eeg_env

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify Python environment
echo "Python environment:"
python --version
echo ""

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

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "Data generation completed successfully!"
    echo "===================================================================="
    echo "Output directory: $OUTPUT_DIR"
    echo "End time: $(date)"
    echo "===================================================================="

    # Print summary statistics
    echo ""
    echo "Summary:"
    echo "  Number of subject directories created:"
    ls -d $OUTPUT_DIR/sub-* 2>/dev/null | wc -l
    echo "  Total disk space used:"
    du -sh $OUTPUT_DIR
    echo "  Number of files per subject (first subject):"
    ls $OUTPUT_DIR/sub-001/ 2>/dev/null | wc -l
    echo ""

else
    echo ""
    echo "===================================================================="
    echo "ERROR: Data generation failed!"
    echo "===================================================================="
    echo "End time: $(date)"
    echo "Please check the error log for details."
    echo "===================================================================="
    exit 1
fi

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
