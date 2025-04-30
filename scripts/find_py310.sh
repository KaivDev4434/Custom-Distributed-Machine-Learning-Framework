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

# List all conda environments and their Python versions
echo "=================== CONDA ENVIRONMENTS ==================="
conda info --envs

echo "=================== SYSTEM PYTHON VERSION ==================="
which python
python --version

echo "=================== CHECKING PYTHON 3.10 AVAILABILITY ==================="
# Check if Python 3.10 is available on the system
if command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is available at: $(which python3.10)"
else
    echo "Python 3.10 is not available as python3.10 command"
fi

echo "=================== CHECKING CONDA ENVIRONMENTS ==================="
# List all conda environments
for env in $(conda env list | grep -v "^#" | awk '{print $1}'); do
    if [ "$env" != "*" ]; then
        echo "Checking environment: $env"
        conda activate $env
        python_path=$(which python)
        python_version=$(python --version)
        echo "  Python path: $python_path"
        echo "  Python version: $python_version"
        conda deactivate
    fi
done

echo "=================== CREATING PYTHON 3.10 ENVIRONMENT ==================="
echo "If you don't see Python 3.10 above, you can create a new environment with:"
echo "conda create -n py310 python=3.10 numpy"
echo "Then use that environment for building and running the code" 