#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Train_Sudoku_Mean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:30:00
#SBATCH --output=outfiles/sudokucnn_dropout_mean_%A.out


module purge
module load 2023
module load Anaconda3/2023.07-2
#module load CUDA/12.4.0

python -m train --config configs/from_baseline/mean/scaler_2_0.yaml
python -m train --config configs/from_baseline/mean/scaler_2_1.yaml
python -m train --config configs/from_baseline/mean/scaler_2_2.yaml
python -m train --config configs/from_baseline/mean/scaler_2_3.yaml
python -m train --config configs/from_baseline/mean/scaler_2_4.yaml
