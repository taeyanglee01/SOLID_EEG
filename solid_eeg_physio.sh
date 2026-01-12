#!/bin/bash
#SBATCH -A m4727_g
#SBATCH -C gpu #&hbm80g
#SBATCH -q shared #preempt #regular #shared #regular, shared,  #! 30 mins is enough so debug
#SBATCH --job-name=SOPh9s50
#SBATCH --output=/pscratch/sd/t/tylee/slurm_outputs/solid/260107_SOLID_Physio_keepratio09_stride50-%A_%a.out
#SBATCH --error=/pscratch/sd/t/tylee/slurm_outputs/solid/260107_SOLID_Physio_keepratio09_stride50-%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00

module load conda
conda activate solid_eeg

cd /pscratch/sd/t/tylee/SOLID_EEG

# seed_list=(41 42 43 44 45)
# seed=${seed_list[$SLURM_ARRAY_TASK_ID]}
# python Physio_1sec_3d.py \
# python Physio_1sec.py \
# python SynEmo_1sec.py \
# python Physio_1sec_percentile.py \
# python Physio_1frame.py \
# python Physio_1frame_modelsearchCNN.py \
python Physio_1frame_modelsearchCNN.py \
    --seed 41 \
    --batch_size 32 \
    --lr 2e-4 \
    --max_lr 4e-5 \
    --min_lr 8e-7 \
    --wd 1e-4 \
    --result_dir /pscratch/sd/t/tylee/SOLID_EEG_RESULT/physio_0112_check6_keepratio07_stride50_Attn_test \
    --dataset_dir /pscratch/sd/t/tylee/Dataset/PhysioNet_200Hz_lowpass40_for_SOLID \
    --squash_tanh True \
    --keep_ratio 0.7 \
    --time_steps 1000 \
    --total_steps 10000 \
    --log_every 200 \
    --eval_every 1000 \
    --save_samples_every 1000 \
    --be_weight 0 \
    --data_scaling_factor 200 \
    --data_segment 50