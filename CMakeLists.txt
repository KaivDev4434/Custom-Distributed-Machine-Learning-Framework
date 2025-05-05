cmake_minimum_required(VERSION 3.8)
project(cuda_model CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets --compiler-options -fPIC")

# Always build with position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add CUDA paths
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# Find pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FIND_RESULT
)

if (PYBIND11_FIND_RESULT EQUAL 0)
    include_directories(${PYBIND11_INCLUDE_DIR})
    message(STATUS "Found pybind11 include directory: ${PYBIND11_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "pybind11 not found. Please install it with pip install pybind11")
endif()

# CUDA source files
set(CUDA_SOURCES
    src/model/Model.cu
)

# C++ source files
set(CPP_SOURCES
    src/model/bindings.cpp
)

# Create CUDA library with explicit PIC flags
cuda_add_library(cuda_kernels STATIC ${CUDA_SOURCES} OPTIONS --compiler-options -fPIC)
set_target_properties(cuda_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Create Python module
add_library(nn_model MODULE ${CPP_SOURCES})
set_target_properties(nn_model PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "nn_model"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Link with Python and CUDA libraries
target_link_libraries(nn_model
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    cuda_kernels
    ${Python3_LIBRARIES}
)

# Set include directories
target_include_directories(nn_model
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDA_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR}
)

# Install rules
install(TARGETS nn_model
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Add test executable (optional)
add_executable(test_model test/test_model.cpp)
target_link_libraries(test_model cuda_kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})