cmake_minimum_required(VERSION 3.20)
project(ParallelComputing LANGUAGES CXX C)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard and architecture
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# CUDA compatibility flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr")

# Find dependencies
find_package(CUDA 12.0 REQUIRED)
find_package(OpenMP REQUIRED)
# Only find MPI if needed
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi")
  find_package(MPI)
endif()
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

# pybind11 - try different approaches to find it
if(NOT pybind11_FOUND)
  find_package(pybind11 CONFIG)
endif()

if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(PYBIND11_INCLUDE_DIR)
    set(pybind11_FOUND TRUE)
    message(STATUS "Found pybind11 include directory: ${PYBIND11_INCLUDE_DIR}")
    include_directories(${PYBIND11_INCLUDE_DIR})
  endif()
endif()

# Add include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CUDA_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)
if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Define the data loader library (exists)
add_library(data_loader SHARED
  src/data_loader/DataLoader.cpp
  src/data_loader/bindings.cpp
)

if(pybind11_FOUND)
  target_link_libraries(data_loader
    ${OpenMP_CXX_LIBRARIES}
    ${Python3_LIBRARIES}
    pybind11::module
  )
else()
  target_link_libraries(data_loader
    ${OpenMP_CXX_LIBRARIES}
    ${Python3_LIBRARIES}
  )
endif()

# Define the model library (exists)
add_library(model SHARED
  src/model/cuda_kernels.cu
  src/model/Model.cpp
  src/model/bindings.cpp
)

target_link_libraries(model
  ${CUDA_LIBRARIES}
  ${Python3_LIBRARIES}
)

if(pybind11_FOUND)
  target_link_libraries(model pybind11::module)
endif()

# Create placeholder mpi compatibility library
# This will allow the benchmarks to still work without actual MPI functionality
add_library(gradient_sync SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model/test_gradient_sync.cu
)

target_link_libraries(gradient_sync
  ${CUDA_LIBRARIES}
  ${Python3_LIBRARIES}
)

if(pybind11_FOUND)
  target_link_libraries(gradient_sync pybind11::module)
endif()

if(MPI_FOUND)
  target_link_libraries(gradient_sync ${MPI_LIBRARIES})
endif()

# Create empty utils library
add_library(utils SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/data_loader/bindings.cpp
)

target_link_libraries(utils
  ${Python3_LIBRARIES}
)

if(pybind11_FOUND)
  target_link_libraries(utils pybind11::module)
endif()

# Set output directory for libraries
set_target_properties(data_loader model gradient_sync utils PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

# Installation rules
install(TARGETS data_loader model gradient_sync utils
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Create Python package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup.py.in
  ${CMAKE_BINARY_DIR}/setup.py @ONLY)

# Post-build: copy libraries to Python module directory
add_custom_command(TARGET data_loader POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/src
  COMMAND ${CMAKE_COMMAND} -E copy 
    $<TARGET_FILE:data_loader> 
    $<TARGET_FILE:model>
    $<TARGET_FILE:gradient_sync>
    $<TARGET_FILE:utils>
    ${CMAKE_BINARY_DIR}/src
)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(MPI_FOUND)
  message(STATUS "  MPI: ${MPI_LIBRARIES}")
else()
  message(STATUS "  MPI: Not used")
endif()
message(STATUS "  Python: ${Python3_LIBRARIES}")
message(STATUS "  OpenMP: ${OpenMP_CXX_LIBRARIES}")
message(STATUS "  Output Directory: ${CMAKE_BINARY_DIR}/lib") 