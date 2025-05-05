#!/bin/bash
set -e

# Build the project
mkdir -p build
cd build
cmake ..
make -j4

# Make sure the Python library is in the PYTHONPATH
export PYTHONPATH=$(pwd)/lib:$PYTHONPATH

# Go back to the project root
cd ..

# Run the benchmark
echo "Running benchmark..."
CUDA_LAUNCH_BLOCKING=1 python benchmarks/benchmark_stage2.py