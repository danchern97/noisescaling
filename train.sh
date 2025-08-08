#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Train_Sudoku_0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:30:00
#SBATCH --output=outfiles/sudokucnn_b_residual_mean_0_%A.out


module purge
module load 2023
module load Anaconda3/2023.07-2
#module load CUDA/12.4.0

python -m train --config configs/from_baseline/mean/scaler_mean_2_0.yaml