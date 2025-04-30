#!/bin/bash

# Check if conda environment name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <conda_env_name> [num_processes]"
    echo "Example: $0 myenv 4"
    exit 1
fi

CONDA_ENV_NAME=$1
NUM_PROCESSES=${2:-4}  # Default to 4 if not provided

# Load required modules
module purge
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4
module load Miniforge3/24.1.2-0

# Set up conda
source "$(dirname $(which conda))"/../etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate $CONDA_ENV_NAME

# Check Python version
echo "Using Python:"
python --version

# Clean build and rebuild
echo "Cleaning and rebuilding project..."
rm -rf build
mkdir -p build
cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make -j$(nproc)
cd ..

# Make sure the build directory is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# Run the simple test first
echo "Running simple MPI test..."
python python/simple_mpi_test.py

# If successful, run the full test with srun
if [ $? -eq 0 ]; then
    echo "Simple test successful, running with srun..."
    
    # Create a wrapper script for srun to ensure conda environment is used
    cat > scripts/temp_wrapper.sh << EOF
#!/bin/bash
source "$(dirname $(which conda))"/../etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME
python \$@
EOF

    chmod +x scripts/temp_wrapper.sh
    
    srun -p gpu \
        -n 1 \
        --ntasks-per-node=$NUM_PROCESSES \
        --qos=standard \
        --account=2025-spring-ds-642-bader-kd454 \
        --gres=gpu:1 \
        --time=1:00:00 \
        scripts/temp_wrapper.sh python/test_mpi.py
    
    # Clean up
    rm -f scripts/temp_wrapper.sh
else
    echo "Simple test failed, please fix the errors before running with srun."
fi

# Deactivate conda environment
conda deactivate 