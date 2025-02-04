#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_reservation
#SBATCH --reservation haicu_stefan
#SBATCH --mem=40G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1

source ~/.bashrc
conda activate sort2learn
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress