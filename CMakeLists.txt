cmake_minimum_required(VERSION 3.10)
project(CustomDLFramework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Add data loader library
add_library(data_loader SHARED
    src/data_loader/DataLoader.cpp
    src/data_loader/bindings.cpp
)

# Set include directories
target_include_directories(data_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(data_loader PRIVATE
    OpenMP::OpenMP_CXX
    pybind11::module
    Python3::Python
)

# Set output directory
set_target_properties(data_loader PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    PREFIX ""
) 