#!/bin/bash

# Load modules
module purge
module load wulver
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4
module load Miniforge3/24.1.2-0
module load CMake

# Set up conda properly
source "$(dirname $(which conda))"/../etc/profile.d/conda.sh

# Activate the Python 3.10 environment
conda activate py310

# Print Python version
echo "Using Python $(python --version 2>&1)"
echo "Environment is ready. You can now build and run your code."
