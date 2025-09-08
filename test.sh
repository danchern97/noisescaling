#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Test_deepbaseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=outfiles/A_test_deepbaseline_%A.out


module purge
module load 2023
module load Anaconda3/2023.07-2

python -m test --config configs/from_baseline/mean/scaler_2_0.yaml
