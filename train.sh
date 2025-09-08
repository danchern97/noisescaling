#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Train_S
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=outfiles/A_deepbaseline_%A.out


module purge
module load 2023
module load Anaconda3/2023.07-2

#python -m train --config configs/from_baseline/mean/scaler_3_4.yaml
#python -m train --config configs/from_baseline/baseline.yaml
#python -m train --config configs/from_baseline/mean/scaler6_4_8.yaml
python -m train --config configs/from_baseline/local_attention/scaler4_6_10.yaml
#python -m train --config configs/from_baseline/conv/scaler_4_7.yaml

#python -m train --config configs/from_baseline/local_attention/scaler_6_8.yaml



# scaler injection points and corresponding dims:
#       if inj_point == 0:
#           dim = 1
#       elif inj_point < 3:
#           dim = 128
#       elif inj_point < 5:
#           dim = 256
#       elif inj_point < 9:
#           dim = 512
#       else:
#           dim = 1024

# for attention aggregator, in_features should be:
#       in_features = dim * H * W at the point of aggregator injection
#
#       if inj_point == 0:
#           in_features = 1 * 81 = 81
#       elif inj_point < 3:
#           in_features = 128 * 9 * 9 = 10368
#       elif inj_point < 5:
#           in_features = 256 * 9 * 9 = 20736
#       elif inj_point < 9:
#           in_features = 512 * 9 * 9 = 41472
#       elif inj_point < 10:
#           in_features = 1024 * 9 * 9 = 82944
#       else:
#           in_features = 729