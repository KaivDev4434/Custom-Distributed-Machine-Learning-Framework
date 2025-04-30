#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpi.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

// Initialize MPI
void init_mpi_wrapper(py::array_t<int> rank_arr, py::array_t<int> size_arr) {
    py::buffer_info rank_buf = rank_arr.request();
    py::buffer_info size_buf = size_arr.request();
    
    int* rank_ptr = static_cast<int*>(rank_buf.ptr);
    int* size_ptr = static_cast<int*>(size_buf.ptr);
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, rank_ptr);
    MPI_Comm_size(MPI_COMM_WORLD, size_ptr);
}

// Synchronize gradients across all processes
py::array_t<float> sync_gradients_wrapper_alt(py::array_t<float> input_gradients) {
    // Instead of modifying the input array, we'll create and return a new array
    
    // Get input data
    py::buffer_info buf = input_gradients.request();
    float* input_ptr = static_cast<float*>(buf.ptr);
    int size = buf.size;
    
    // Get rank for debug
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "ALT [Rank " << rank << "]: First element before sync: " << input_ptr[0] << std::endl;
    
    // Create a vector to store the result
    std::vector<float> result(size);
    
    // Perform AllReduce to sum gradients
    MPI_Allreduce(input_ptr, result.data(), size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    std::cout << "ALT [Rank " << rank << "]: First element after reduce (before avg): " << result[0] << std::endl;
    
    // Get world size for averaging
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Average the results
    for (int i = 0; i < size; i++) {
        result[i] /= world_size;
    }
    
    std::cout << "ALT [Rank " << rank << "]: First element after sync: " << result[0] << std::endl;
    
    // Create a new NumPy array with the result
    py::array_t<float> output_gradients = py::array_t<float>(buf.size);
    py::buffer_info out_buf = output_gradients.request();
    float* output_ptr = static_cast<float*>(out_buf.ptr);
    
    // Copy the data to the output array
    for (int i = 0; i < size; i++) {
        output_ptr[i] = result[i];
    }
    
    return output_gradients;
}

// Finalize MPI
void finalize_mpi_wrapper() {
    MPI_Finalize();
}

// Get the world size
int get_world_size_wrapper() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
}

// Get the world rank
int get_world_rank_wrapper() {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return world_rank;
}

PYBIND11_MODULE(mpi_wrapper_alt, m) {
    m.doc() = "Alternative Python bindings for MPI gradient synchronization";
    
    m.def("init_mpi", &init_mpi_wrapper, "Initialize MPI");
    m.def("sync_gradients", &sync_gradients_wrapper_alt, "Synchronize gradients across processes");
    m.def("finalize_mpi", &finalize_mpi_wrapper, "Finalize MPI");
    m.def("get_world_size", &get_world_size_wrapper, "Get the number of processes");
    m.def("get_world_rank", &get_world_rank_wrapper, "Get the current process rank");
} 