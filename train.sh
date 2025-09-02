#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Train_Sudoku_Attention
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=outfiles/sudokucnn_att_deepbaseline_%A.out


module purge
module load 2023
module load Anaconda3/2023.07-2
#module load CUDA/12.4.0

python -m train --config configs/from_baseline/attention/scaler_2_0.yaml