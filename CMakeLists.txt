cmake_minimum_required(VERSION 3.10)
project(CustomDLFramework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

# CUDA compatibility configuration
# Force CUDA to use a compatible standard with newer GCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# Add flags to handle newer GCC compatibility issues
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-float-conversion")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
# Important flag to work around the __float128 issues
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_NO_HALF_CONVERSIONS__")

# Set CUDA architecture - using a wide range for compatibility
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Add data loader library
add_library(data_loader SHARED
    src/data_loader/DataLoader.cpp
    src/data_loader/bindings.cpp
)

# Set data loader properties
target_link_libraries(data_loader PRIVATE
    OpenMP::OpenMP_CXX
    pybind11::module
    Python3::Python
)

set_target_properties(data_loader PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    PREFIX ""
)

# First build without the CUDA components to ensure data_loader works
# Add model library conditionally based on CUDA availability
if(CUDA_FOUND)
    message(STATUS "CUDA found, building nn_model library")
    
    add_library(nn_model SHARED
        src/model/Model.cpp
        src/model/cuda_kernels.cu
        src/model/bindings.cpp
    )

    # Set model properties
    target_link_libraries(nn_model PRIVATE
        ${CUDA_LIBRARIES}
        pybind11::module
        Python3::Python
    )

    set_target_properties(nn_model PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
        PREFIX ""
    )
else()
    message(STATUS "CUDA not found or has issues, skipping nn_model library")
endif() 