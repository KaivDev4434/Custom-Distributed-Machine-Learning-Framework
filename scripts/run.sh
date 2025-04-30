#!/bin/bash
#SBATCH --job-name=test_data_loader
#SBATCH --output=test_data_loader_%j.out
#SBATCH --error=test_data_loader_%j.err
#SBATCH --account=2025-spring-ds-642-bader-kd454
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
# #SBATCH --mem=4G

#load modules
#module purge
#module load wulver
#module load GCC
#module load CMake
module load Miniforge3

#activate env
conda activate apc_proj

# Run your script
python python/test_data_loader.py
