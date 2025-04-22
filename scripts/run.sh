#!/bin/bash
#SBATCH --job-name=test_data_loader
#SBATCH --output=test_data_loader_%j.out
#SBATCH --error=test_data_loader_%j.err
#SBATCH --account=2025-spring-ds-642-bader-kd454
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Run your script
python python/test_data_loader.py
