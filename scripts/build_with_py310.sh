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
