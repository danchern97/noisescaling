#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:25:00
#SBATCH --output=outfiles/env_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

#conda create -n agi python=3.8.2
source activate agi

python --version 

pip install -r requirements.txt

pip install "protobuf<4.0"

pip uninstall -y torch torchvision torchaudio

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
