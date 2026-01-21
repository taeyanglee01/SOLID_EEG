#!/bin/bash
#SBATCH -A m4727_g
#SBATCH -C gpu #&hbm80g
#SBATCH -q shared #preempt #regular #shared #regular, shared,  #! 30 mins is enough so debug
#SBATCH --job-name=SOPhMLP
#SBATCH --output=/pscratch/sd/t/tylee/slurm_outputs/solid/260118_SOLID_Physio_keepratio07_stride50_MLPMixer-%A_%a.out
#SBATCH --error=/pscratch/sd/t/tylee/slurm_outputs/solid/260118_SOLID_Physio_keepratio07_stride50_MLPMixer-%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 6:00:00

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
# python Physio_1frame_modelsearch_factorized.py \
# python Physio_1frame_modelsearchCNN_3D.py \
python Physio_1frame_modelsearchCNN_1inst.py \
    --seed 41 \
    --batch_size 32 \
    --dmodel 128 \
    --lr 2e-4 \
    --max_lr 4e-5 \
    --min_lr 8e-7 \
    --wd 1e-4 \
    --result_dir /pscratch/sd/t/tylee/SOLID_EEG_RESULT/physio_0121_check4OneInst_keepratio10_stride10_50000iter \
    --dataset_dir /pscratch/sd/t/tylee/Dataset/PhysioNet_200Hz_lowpass40_for_SOLID \
    --squash_tanh True \
    --keep_ratio 1.0 \
    --time_steps 1000 \
    --total_steps 50000 \
    --log_every 200 \
    --eval_every 1000 \
    --save_samples_every 1000 \
    --be_weight 0 \
    --data_scaling_factor 200 \
    --data_segment 10 \
    --use_wavelet_loss False \
    --use_spectral_loss False \
    --use_perceptual_loss False