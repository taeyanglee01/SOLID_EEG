#!/bin/bash
#SBATCH --job-name=synth_to_lmdb
#SBATCH --output=logs/preproc_lmdb_%j.out
#SBATCH --error=logs/preproc_lmdb_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --account=m4239
#SBATCH --constraint=cpu
#SBATCH --qos=regular

# ==============================================================================
# Preprocess Synthetic EEG and Convert to LMDB
# ==============================================================================

# Configuration
INPUT_DIR="/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_data"
OUTPUT_LMDB="/pscratch/sd/t/tylee/Dataset/Synthetic_EEG_Motor_Imagery_200Hz"
TASKS="motor_imagery"  # or "all", "seizure", "p300", "emotion", or comma-separated
TARGET_SFREQ=200
MAP_SIZE=50000000000  # 50GB

# ==============================================================================
# Setup
# ==============================================================================

mkdir -p logs

echo "===================================================================="
echo "Synthetic EEG Preprocessing to LMDB"
echo "===================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "===================================================================="
echo ""

# Environment setup
source activate base

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Configuration:"
echo "  Input directory: $INPUT_DIR"
echo "  Output LMDB: $OUTPUT_LMDB"
echo "  Tasks: $TASKS"
echo "  Target sampling frequency: $TARGET_SFREQ Hz"
echo "  LMDB map size: $MAP_SIZE bytes"
echo ""

# ==============================================================================
# Run Preprocessing
# ==============================================================================

python preproc_synthetic_to_lmdb.py \
    --input_dir $INPUT_DIR \
    --output_lmdb $OUTPUT_LMDB \
    --tasks $TASKS \
    --target_sfreq $TARGET_SFREQ \
    --map_size $MAP_SIZE

# ==============================================================================
# Check Results
# ==============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "Preprocessing completed successfully!"
    echo "===================================================================="
    echo "Output LMDB: $OUTPUT_LMDB"

    # Check LMDB size
    echo ""
    echo "LMDB database info:"
    du -sh $OUTPUT_LMDB

    echo ""
    echo "End time: $(date)"
    echo "===================================================================="
else
    echo ""
    echo "===================================================================="
    echo "ERROR: Preprocessing failed!"
    echo "===================================================================="
    exit 1
fi

exit 0
