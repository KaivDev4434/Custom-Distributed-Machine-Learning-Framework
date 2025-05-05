cmake_minimum_required(VERSION 3.10)
project(test_model CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)

# Add test executable
add_executable(test_model test_model.cpp ../src/model/Model.cpp ../src/model/cuda_kernels.cu)
target_include_directories(test_model PRIVATE ..)
target_link_libraries(test_model ${CUDA_LIBRARIES})
set_target_properties(test_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 