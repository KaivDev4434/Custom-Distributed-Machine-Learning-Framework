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

# Find conda executable and print its location
CONDA_EXE=$(which conda)
echo "Conda executable: $CONDA_EXE"

# Use the correct path for conda.sh based on user's environment
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"

echo "Checking for conda.sh at: $CONDA_SH"
if [ -f "$CONDA_SH" ]; then
    echo "Found conda.sh, sourcing it"
    source "$CONDA_SH"
else
    echo "ERROR: conda.sh not found at $CONDA_SH"
    echo "Please locate your conda.sh file with:"
    echo "find /mmfs1 -name conda.sh 2>/dev/null"
    exit 1
fi

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
pip install numpy
pip install pybind11

# Verify pybind11 installation
echo "Checking pybind11 installation:"
pip list | grep pybind11

# Install pybind11 development files for CMake
echo "Installing pybind11 development files..."
mkdir -p ~/pybind11_build
cd ~/pybind11_build
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
cd ~/pybind11_build

# Check if pybind11 was installed correctly
if [ -d "$CONDA_PREFIX/share/cmake/pybind11" ]; then
    echo "pybind11 installed successfully"
else
    echo "pybind11 installation might have failed"
fi

cd - # Return to original directory

# Clean build and rebuild with pybind11 path
echo "Cleaning and rebuilding project..."
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

echo "Build completed. Check for any errors above."
echo "To test, run: ./scripts/simple_test.sh"

# Create a test script
cat > scripts/simple_test.sh << EOF
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

# Activate the Python 3.10 environment
conda activate py310

# Set PYTHONPATH
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/build/python

# Run the simple test
echo "Running simple MPI test..."
python python/simple_mpi_test.py

EOF

chmod +x scripts/simple_test.sh 