#!/bin/bash
# Build script for compiling the project with the required modules

# Load wulver first (required on this HPC system)
module purge
module load wulver

# Load Miniforge3 to access conda
module load Miniforge3

# Load required modules - using compatible GCC version
module load foss/2022b
module load CUDA/12.0.0
module load GCC/12.2.0
module load CMake/3.24.3  # CMake version compatible with foss/2022b

# Activate conda for Python packages
source $(conda info --base)/etc/profile.d/conda.sh
conda activate apc_proj

echo "=== Build Environment ==="
echo "Using modules:"
echo "  CUDA: $(nvcc --version | head -n 1)"
echo "  GCC: $(gcc --version | head -n 1)"
echo "  CMake: $(cmake --version | head -n 1)"
echo "Using conda environment: apc_proj"
echo "  Python: $(python --version)"
echo "========================="

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Clean any previous build artifacts
rm -rf CMakeCache.txt CMakeFiles/

# Configure with CMake with CUDA workarounds
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DCMAKE_CUDA_FLAGS="-D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr"

# Build the project
echo "Building the project..."
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "You can now run the benchmarks with: sbatch scripts/launch_benchmarks.slurm"
else
    echo "Build failed. Please check the output for errors."
    exit 1
fi

# Create necessary directories
cd ..
mkdir -p data/mnist
mkdir -p models
mkdir -p results

# Deactivate conda environment
conda deactivate

echo "Build completed successfully!"
echo "The libraries are available in build/lib/" 