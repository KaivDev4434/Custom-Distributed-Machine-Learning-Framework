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

# Activate the conda environment
echo "Activating conda environment: apc_proj"
conda activate apc_proj

# Check Python version and location
echo "Python version:"
python --version
echo "Python location: $(which python)"

# Check conda environment details
echo "Conda environment details:"
conda info --envs

# Check if numpy is available
echo "Checking for numpy:"
pip list | grep numpy

# Try to import numpy directly
echo "Testing numpy import directly:"
python -c "import numpy; print('NumPy version:', numpy.__version__)"

echo "If you see a NumPy version above, the environment is correctly set up"

# Try to run the simple test script
echo "Attempting to run the simple MPI test:"
python python/simple_mpi_test.py 