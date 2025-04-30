cmake_minimum_required(VERSION 3.10)
project(gradient_sync_test CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Important: Use the correct way to pass NVCC flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -allow-unsupported-compiler)

# Add the test executable
cuda_add_executable(test_gradient_sync
    test_gradient_sync.cu
    cuda_kernels.cu
)

# Link CUDA libraries
target_link_libraries(test_gradient_sync ${CUDA_LIBRARIES})

# Set CUDA architecture
set_property(TARGET test_gradient_sync
    PROPERTY CUDA_ARCHITECTURES 70 75 80)  # Support for multiple GPU architectures