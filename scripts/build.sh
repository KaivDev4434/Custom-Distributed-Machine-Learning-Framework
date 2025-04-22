#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make -j$(nproc)

# Create data directory if it doesn't exist
mkdir -p ../data/mnist

echo "Build completed successfully!"
echo "The data_loader module is available in build/python/" 