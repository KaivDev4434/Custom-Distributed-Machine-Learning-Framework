#!/bin/bash

# Load the correct Python environment
echo "Loading correct Python environment..."
module purge
module load wulver
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4
module load Miniforge3/24.1.2-0
module load CMake

# Set up conda properly - this is important!
echo "Setting up conda..."
source "$(dirname $(which conda))"/../etc/profile.d/conda.sh

# Activate the conda environment
echo "Activating conda environment: apc_proj"
conda activate apc_proj

# Verify conda environment and packages
echo "Conda environment info:"
conda info
echo "Checking for numpy:"
pip list | grep numpy

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
    
    # Check if number of processes was provided
    NUM_PROCESSES=${1:-4}  # Default to 4 if not provided
    
    # Create a wrapper script for srun that activates the conda environment
    echo "Creating conda wrapper script for srun..."
    cat > scripts/conda_wrapper.sh << EOF
#!/bin/bash
source "$(dirname $(which conda))"/../etc/profile.d/conda.sh
conda activate apc_proj
python "\$@"
EOF

    chmod +x scripts/conda_wrapper.sh
    
    srun -p gpu \
        -n 1 \
        --ntasks-per-node=$NUM_PROCESSES \
        --qos=standard \
        --account=2025-spring-ds-642-bader-kd454 \
        --gres=gpu:1 \
        --time=1:00:00 \
        scripts/conda_wrapper.sh python/test_mpi.py
        
    # Clean up
    rm -f scripts/conda_wrapper.sh
else
    echo "Simple test failed, please fix the errors before running with srun."
fi 
 