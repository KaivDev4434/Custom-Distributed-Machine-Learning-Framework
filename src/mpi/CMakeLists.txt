cmake_minimum_required(VERSION 3.10)
project(mpi_wrapper)

# Set policy to use _ROOT variables
cmake_policy(SET CMP0074 NEW)

# Set Python root to conda environment
set(PYTHON_ROOT "/project/kjc59/kd454/envs/apc_proj")

# Find Python and its components
find_package(Python 3.13.3 EXACT REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(MPI REQUIRED)

# Add the MPI wrapper source files
add_library(mpi_wrapper SHARED
    mpi_wrapper.cpp
    gradient_sync.c
)

# Link against MPI and pybind11
target_link_libraries(mpi_wrapper
    PRIVATE
    MPI::MPI_CXX
    pybind11::module
    Python::Python
)

# Set output directory to the Python directory
set_target_properties(mpi_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
    PREFIX ""
    SUFFIX ".so"
)

# Add include directories
target_include_directories(mpi_wrapper
    PRIVATE
    ${MPI_CXX_INCLUDE_PATH}
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Set Python-specific compiler flags
target_compile_options(mpi_wrapper
    PRIVATE
    -fPIC
    -DPYBIND11_PYTHON_VERSION=3.13.3
)

# Set Python-specific linker flags
target_link_options(mpi_wrapper
    PRIVATE
    -L${PYTHON_ROOT}/lib
    -lpython3.13
) 