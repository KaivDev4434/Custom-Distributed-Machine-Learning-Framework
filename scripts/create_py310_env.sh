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

# Check if the py310 environment already exists
if conda env list | grep -q "py310"; then
    echo "Python 3.10 environment 'py310' already exists"
else
    echo "Creating Python 3.10 environment 'py310'..."
    conda create -y -n py310 python=3.10 numpy
fi

# Activate the Python 3.10 environment
echo "Activating Python 3.10 environment..."
conda activate py310

# Verify Python version
echo "Python version:"
python --version

# Install required packages
echo "Installing required packages..."
pip install numpy pybind11

echo "Environment is ready. You can now build and run your code with Python 3.10."
echo "To use this environment, run: source scripts/use_py310.sh"

# Create a script to use this environment
cat > scripts/use_py310.sh << 'EOF'
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
EOF

chmod +x scripts/use_py310.sh

# Create a build script for Python 3.10
cat > scripts/build_with_py310.sh << 'EOF'
#!/bin/bash

# Source the Python 3.10 environment
source scripts/use_py310.sh

# Clean build and rebuild
echo "Cleaning and rebuilding project..."
rm -rf build
mkdir -p build
cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make -j$(nproc)
cd ..

echo "Build completed. MPI wrapper has been compiled with Python 3.10."
echo "To test, run: ./scripts/test_with_py310.sh"
EOF

chmod +x scripts/build_with_py310.sh

# Create a test script for Python 3.10
cat > scripts/test_with_py310.sh << 'EOF'
#!/bin/bash

# Source the Python 3.10 environment
source scripts/use_py310.sh

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
    cat > scripts/py310_wrapper.sh << 'INNEREOF'
#!/bin/bash
source "$(dirname $(which conda))"/../etc/profile.d/conda.sh
conda activate py310
python "$@"
INNEREOF

    chmod +x scripts/py310_wrapper.sh
    
    echo "Running MPI test with $NUM_PROCESSES processes..."
    srun -p gpu \
        -n 1 \
        --ntasks-per-node=$NUM_PROCESSES \
        --qos=standard \
        --account=2025-spring-ds-642-bader-kd454 \
        --gres=gpu:1 \
        --time=1:00:00 \
        scripts/py310_wrapper.sh python/test_mpi.py
        
    # Clean up
    rm -f scripts/py310_wrapper.sh
else
    echo "Simple test failed, please fix the errors before running with srun."
fi
EOF

chmod +x scripts/test_with_py310.sh

echo "All scripts created. Follow these steps:"
echo "1. ./scripts/build_with_py310.sh    # Build with Python 3.10"
echo "2. ./scripts/test_with_py310.sh     # Test with Python 3.10" 