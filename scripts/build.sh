#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make -j$(nproc)

# Create necessary directories
cd ..
mkdir -p data/mnist
mkdir -p models
mkdir -p results

echo "Build completed successfully!"
echo "The data_loader and nn_model modules are available in build/python/" 