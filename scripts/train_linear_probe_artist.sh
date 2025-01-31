#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=TrainProbes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=TrainProbes_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Your job starts in the directory where you call sbatch
cd $HOME/Discover-then-Name
# Activate your environment
source activate fact_env

# python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 0 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset cifar10 --use_wandb
python scripts/train_linear_probe_artist.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 0 --val_freq 1 --probe_lr 1e-2  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset cifar100 --use_wandb
#python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 1 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset places365 --use_wandb
#python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 1 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset full-imagenet --use_wandb

# python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_ViT-B/16 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 1 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset cifar10 --use_wandb
# python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_ViT-B/16 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 1 --val_freq 1 --probe_lr 1e-2  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset cifar100 --use_wandb
# python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_ViT-B/16 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 1 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset places365 --use_wandb
# python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_ViT-B/16 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 1 --val_freq 1 --probe_lr 1e-3  --probe_sparsity_loss_lambda 1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset full-imagenet --use_wandb