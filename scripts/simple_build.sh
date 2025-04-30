#!/bin/bash

# Load modules with explicit paths
module purge
module load wulver
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4
module load Miniforge3/24.1.2-0
module load CMake

# Use the correct path for conda.sh
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"

# Activate the py310 environment
echo "Activating Python 3.10 environment..."
conda activate py310

# Verify Python version
echo "Python version:"
python --version

# Install required packages
echo "Installing required packages..."
pip install numpy
pip install pybind11

# Create a simple pybind11 config file
mkdir -p build/pybind11
cat > build/pybind11/pybind11Config.cmake << EOF
# Generated pybind11 config file
set(pybind11_INCLUDE_DIRS "$(python -c "import pybind11; print(pybind11.get_include())")")
set(pybind11_LIBRARIES "")
set(pybind11_FOUND TRUE)
EOF

echo "Created simple pybind11 config at build/pybind11/pybind11Config.cmake"

# Clean and rebuild project
echo "Rebuilding project..."
cd $(pwd)  # Make sure we're in the project root
mkdir -p build/python
cd build

# Configure with explicit pybind11 path
echo "Running CMake..."
cmake -DPython3_EXECUTABLE=$(which python) \
      -Dpybind11_DIR=$(pwd)/pybind11 \
      ..

# Build
echo "Building..."
make -j$(nproc)
cd ..

echo "Build completed. Check for any errors above."

# Create a simple test script
cat > scripts/run_test.sh << 'EOF'
#!/bin/bash

# Load modules and activate conda environment
module purge
module load wulver
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4
module load Miniforge3/24.1.2-0
module load CMake

# Use the correct path for conda.sh
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"

# Activate the Python 3.10 environment
conda activate py310

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# Run the simple test
echo "Running simple MPI test..."
python python/simple_mpi_test.py

# If successful, run the full test with srun
if [ $? -eq 0 ]; then
    echo "Simple test successful, running with srun..."
    
    # Check if number of processes was provided
    NUM_PROCESSES=${1:-4}  # Default to 4 if not provided
    
    # Create a wrapper script for srun
    cat > scripts/srun_wrapper.sh << 'INNEREOF'
#!/bin/bash
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py310
export PYTHONPATH=$PYTHONPATH
python "$@"
INNEREOF

    chmod +x scripts/srun_wrapper.sh
    
    echo "Running MPI test with $NUM_PROCESSES processes..."
    srun -p gpu \
        -n 1 \
        --ntasks-per-node=$NUM_PROCESSES \
        --qos=standard \
        --account=2025-spring-ds-642-bader-kd454 \
        --gres=gpu:1 \
        --time=1:00:00 \
        scripts/srun_wrapper.sh python/test_mpi.py
        
    # Clean up
    rm -f scripts/srun_wrapper.sh
else
    echo "Simple test failed, please fix the errors before running with srun."
fi
EOF

chmod +x scripts/run_test.sh

echo "All done! To test, run: ./scripts/run_test.sh" 