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

# Create a CMakeLists.txt for the alternative implementation
echo "Creating CMake file for alternative implementation..."

cat > CMakeLists.txt.alt << EOF
cmake_minimum_required(VERSION 3.10)
project(MpiWrapperAlt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(MPI REQUIRED)

# Manually set pybind11 include directory
execute_process(
    COMMAND "\${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Include directories
include_directories(
    \${CMAKE_CURRENT_SOURCE_DIR}/src
    \${Python3_INCLUDE_DIRS}
    \${PYBIND11_INCLUDE_DIR}
    \${MPI_CXX_INCLUDE_PATH}
)

# Add MPI wrapper library
add_library(mpi_wrapper_alt SHARED
    src/mpi/mpi_wrapper_alt.cpp
)

# Set MPI wrapper properties
target_link_libraries(mpi_wrapper_alt PRIVATE
    \${MPI_CXX_LIBRARIES}
    \${Python3_LIBRARIES}
)

# Add include directories specifically for MPI wrapper
target_include_directories(mpi_wrapper_alt PRIVATE
    \${MPI_CXX_INCLUDE_PATH}
)

set_target_properties(mpi_wrapper_alt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "\${CMAKE_BINARY_DIR}/python"
    PREFIX ""
)
EOF

# Create build directory for the alternative implementation
echo "Building alternative implementation..."
mkdir -p build_alt/python
cd build_alt

# Configure and build
cmake -DPython3_EXECUTABLE=$(which python) -C ../CMakeLists.txt.alt ..
make -j$(nproc)

cd ..

# Set PYTHONPATH to include both build directories
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python:$(pwd)/build_alt/python

# Run the alternative test
echo "Running the alternative MPI test..."

# Number of processes to use
NUM_PROCESSES=${1:-4}  # Default to 4 if not provided

echo "Testing alternative MPI gradient synchronization with $NUM_PROCESSES processes..."

# Create a wrapper script for mpirun
cat > scripts/mpi_alt_wrapper.sh << 'EOF'
#!/bin/bash
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py310

# Add build directories to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $0))/build/python:$(dirname $(dirname $0))/build_alt/python

exec python "$@"
EOF

chmod +x scripts/mpi_alt_wrapper.sh

# Run the test with mpirun
mpirun -np $NUM_PROCESSES scripts/mpi_alt_wrapper.sh python/test_alt_mpi.py

# Clean up
rm -f scripts/mpi_alt_wrapper.sh 