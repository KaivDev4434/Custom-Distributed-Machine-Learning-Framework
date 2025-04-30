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

# Use the correct path for conda.sh
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"

# Activate the apc_proj environment
echo "Activating apc_proj environment..."
conda activate apc_proj

# Check Python version
echo "Current Python version in apc_proj:"
python --version

# Prompt for confirmation
read -p "Do you want to install Python 3.10 in this environment? This might modify packages. (y/n) " answer
if [[ $answer != "y" && $answer != "Y" ]]; then
    echo "Cancelled. Exiting without changes."
    exit 0
fi

# Install Python 3.10
echo "Installing Python 3.10 in apc_proj environment..."
conda install -y python=3.10

# Check the new Python version
echo "New Python version:"
python --version

# Install required packages
echo "Installing required packages..."
pip install numpy
pip install pybind11

# Install pybind11 development files
echo "Installing pybind11 development files..."
mkdir -p ~/pybind11_apc_proj
cd ~/pybind11_apc_proj
if [ ! -d "pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git
    cd pybind11
    mkdir -p build && cd build
    cmake -DPYBIND11_TEST=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make -j$(nproc)
    make install
else
    echo "pybind11 already cloned"
    cd pybind11/build
    make install
fi

# Return to original directory
cd - 

# Build the project with apc_proj environment
echo "Building project with apc_proj environment..."
rm -rf build
mkdir -p build
cd build

# Configure with explicit pybind11 path
cmake -DPython3_EXECUTABLE=$(which python) \
      -Dpybind11_DIR=$CONDA_PREFIX/share/cmake/pybind11 \
      ..

# Build
make -j$(nproc)
cd ..

echo "Build completed. Now you can test the MPI gradient sync:"
echo "./scripts/test_apc_proj.sh"

# Create a test script
cat > scripts/test_apc_proj.sh << EOF
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
CONDA_SH="\${CONDA_DIR}/etc/profile.d/conda.sh"
source "\$CONDA_SH"

# Activate the apc_proj environment
conda activate apc_proj

# Set PYTHONPATH
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/build/python

# Run the simple test
echo "Running simple MPI test..."
python python/simple_mpi_test.py

# If successful, run the full test with srun
if [ \$? -eq 0 ]; then
    echo "Simple test successful, running with srun..."
    
    # Check if number of processes was provided
    NUM_PROCESSES=\${1:-4}  # Default to 4 if not provided
    
    # Create a wrapper script for srun
    cat > scripts/srun_apc_proj.sh << 'INNEREOF'
#!/bin/bash
source "${CONDA_SH}"
conda activate apc_proj
export PYTHONPATH=\$PYTHONPATH
python "\$@"
INNEREOF

    chmod +x scripts/srun_apc_proj.sh
    
    echo "Running MPI test with \$NUM_PROCESSES processes..."
    srun -p gpu \\
        -n 1 \\
        --ntasks-per-node=\$NUM_PROCESSES \\
        --qos=standard \\
        --account=2025-spring-ds-642-bader-kd454 \\
        --gres=gpu:1 \\
        --time=1:00:00 \\
        scripts/srun_apc_proj.sh python/test_mpi.py
        
    # Clean up
    rm -f scripts/srun_apc_proj.sh
else
    echo "Simple test failed, please fix the errors before running with srun."
fi
EOF

chmod +x scripts/test_apc_proj.sh 