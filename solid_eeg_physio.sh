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

module load conda
conda activate solid_eeg

cd /pscratch/sd/t/tylee/SOLID_EEG

# seed_list=(41 42 43 44 45)
# seed=${seed_list[$SLURM_ARRAY_TASK_ID]}
# python Physio_1sec_3d.py \
# python Physio_1sec.py \
python Physio_1sec.py \
    --seed 41 \
    --batch_size 128 \
    --lr 2e-4 \
    --max_lr 4e-3 \
    --min_lr 8e-5 \
    --wd 1e-4 \
    --result_dir /pscratch/sd/t/tylee/SOLID_EEG_RESULT/physio_1228_check10 \
    --squash_tanh False \
    --time_steps 1000 \
    --total_steps 10000 \
    --log_every 200 \
    --eval_every 1000 \
    --save_samples_every 1000 \
    --be_weight 0