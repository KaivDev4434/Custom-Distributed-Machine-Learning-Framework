#!/bin/bash

# Create build directory
mkdir -p build_dataloader
cd build_dataloader

# Configure and build without CUDA
cmake -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE ..
make -j$(nproc) data_loader

# Create necessary directories
cd ..
mkdir -p data/mnist
mkdir -p models
mkdir -p results

echo "Build completed successfully!"
echo "The data_loader module is available in build_dataloader/python/" 