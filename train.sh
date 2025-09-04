#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Train_S
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=outfiles/A_sudokucnn_mean_deepbaseline_%A.out


module purge
module load 2023
module load Anaconda3/2023.07-2

python -m train --config configs/from_baseline/mean/scaler_2_0.yaml
