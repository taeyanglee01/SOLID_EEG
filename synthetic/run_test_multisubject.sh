#!/bin/bash

# ==============================================================================
# Quick Test Script for Multi-Subject EEG Data Generation
# ==============================================================================
# This script generates a small test dataset with 3 subjects to verify
# that everything is working correctly before running a large-scale generation.
# ==============================================================================

echo "===================================================================="
echo "Multi-Subject EEG Data Generation - Quick Test"
echo "===================================================================="
echo "This will generate data for 3 subjects with short durations."
echo "Estimated time: ~2-3 minutes"
echo "===================================================================="
echo ""

# Configuration for quick test
N_SUBJECTS=3
OUTPUT_DIR="./test_multisubject_output"
N_CHANNELS=64
SFREQ=250

# Shorter durations for quick testing
MOTOR_DURATION=30      # 30 seconds instead of 120
SEIZURE_DURATION=30    # 30 seconds instead of 120
P300_DURATION=60       # 60 seconds instead of 600
EMOTION_DURATION=30    # 30 seconds instead of 240

# Test with all tasks
TASKS="all"

# ==============================================================================
# Setup
# ==============================================================================

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Print configuration
echo "Test Configuration:"
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

# Ask for confirmation
read -p "Continue with test generation? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Test cancelled."
    exit 1
fi

# ==============================================================================
# Run Test Generation
# ==============================================================================

echo ""
echo "Starting test generation..."
echo ""

# Record start time
START_TIME=$(date +%s)

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

# Record end time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# ==============================================================================
# Check Results
# ==============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "Test generation completed successfully!"
    echo "===================================================================="
    echo ""
    echo "Summary:"
    echo "  Elapsed time: $ELAPSED_TIME seconds"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""

    # Count subjects
    N_SUBJECTS_CREATED=$(ls -d $OUTPUT_DIR/sub-* 2>/dev/null | wc -l)
    echo "  Number of subjects created: $N_SUBJECTS_CREATED"

    # Show first subject's files
    echo ""
    echo "Files in first subject directory (sub-001):"
    ls -lh $OUTPUT_DIR/sub-001/ 2>/dev/null | grep -v "^total" | awk '{print "    " $9 " (" $5 ")"}'

    # Show disk usage
    echo ""
    echo "  Total disk space used:"
    du -sh $OUTPUT_DIR

    # Calculate average per subject
    TOTAL_SIZE=$(du -sb $OUTPUT_DIR | awk '{print $1}')
    AVG_SIZE=$((TOTAL_SIZE / N_SUBJECTS_CREATED))
    AVG_SIZE_MB=$((AVG_SIZE / 1024 / 1024))
    echo "  Average per subject: ~${AVG_SIZE_MB} MB"

    echo ""
    echo "===================================================================="
    echo "Test passed! You can now run the full generation."
    echo ""
    echo "To generate full dataset with 100 subjects:"
    echo "  bash run_multisubject_generation.sh"
    echo ""
    echo "Or submit as SLURM job:"
    echo "  sbatch run_multisubject_generation.sh"
    echo ""
    echo "To clean up test data:"
    echo "  rm -rf $OUTPUT_DIR"
    echo "===================================================================="

else
    echo ""
    echo "===================================================================="
    echo "ERROR: Test generation failed!"
    echo "===================================================================="
    echo "Please check the error messages above."
    echo "===================================================================="
    exit 1
fi

exit 0
