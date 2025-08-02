#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Train_Sudoku
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=outfiles/sudokucnn_mean_2_2_orth02_%A.out


module purge
module load 2023
module load Anaconda3/2023.07-2
#module load CUDA/12.4.0

python -m train --config configs/orth_loss/scaler_mean_2_2_orth02.yaml
