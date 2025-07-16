#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=TestModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:15:00
#SBATCH --output=outfiles/test_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0

source activate agi

export CUDA_VISIBLE_DEVICES=0


python test_model.py problem.model.model_path="/gpfs/home5/mmazuryk1/AGI/deep-thinking//outputs/training_default/training-worldly-Latangela"