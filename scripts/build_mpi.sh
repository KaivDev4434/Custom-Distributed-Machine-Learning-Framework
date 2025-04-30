#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Clean previous build
rm -rf *

# Set Python root to conda environment
export PYTHON_ROOT="/project/kjc59/kd454/envs/apc_proj"

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_ROOT=$PYTHON_ROOT \
    -DPYBIND11_PYTHON_VERSION=3.13.3

# Build the project
make -j$(nproc)

# Copy the built module to the Python directory
cp python/mpi_wrapper*.so ../python/

echo "MPI wrapper module built successfully!" 