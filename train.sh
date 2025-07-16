#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=TrainModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:25:00
#SBATCH --output=outfiles/train_noise_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0

source activate agi

export CUDA_VISIBLE_DEVICES=0

#python train_model.py 

#python train_model.py problem.hyp.alpha=1 problem.hyp.lr=0.0001 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation

#python train_model.py problem.hyp.alpha=1 problem/model=ff_net_recall_1d problem=prefix_sums name=prefix_sums_ablation

python train_model.py problem/model=resnet_1d problem=prefix_sums name=prefix_sums_ablation

