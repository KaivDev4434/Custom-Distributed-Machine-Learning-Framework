#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpi.h>
#include <cstring> // for memcpy
#include <iostream> // for debug prints

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
void sync_gradients_wrapper(py::array_t<float> gradients) {
    py::buffer_info buf = gradients.request();
    float* ptr = static_cast<float*>(buf.ptr);
    int size = buf.size;
    
    // Debug prints
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "DEBUG [Rank " << rank << "]: Starting sync_gradients with " << size << " elements" << std::endl;
    std::cout << "DEBUG [Rank " << rank << "]: First element before sync: " << ptr[0] << std::endl;
    
    // Create a buffer for receiving the sum
    float* recv_buffer = new float[size];
    
    // Check if our input and output buffers are different
    std::cout << "DEBUG [Rank " << rank << "]: Input ptr = " << ptr << ", output ptr = " << recv_buffer << std::endl;
    
    // Perform AllReduce to sum gradients
    int result = MPI_Allreduce(ptr, recv_buffer, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    // Check MPI_Allreduce result
    if (result != MPI_SUCCESS) {
        std::cerr << "DEBUG [Rank " << rank << "]: MPI_Allreduce failed with error code " << result << std::endl;
    } else {
        std::cout << "DEBUG [Rank " << rank << "]: MPI_Allreduce succeeded" << std::endl;
        std::cout << "DEBUG [Rank " << rank << "]: First element after reduce (before avg): " << recv_buffer[0] << std::endl;
    }
    
    // Average the gradients and copy back to the original array
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    for (int i = 0; i < size; i++) {
        ptr[i] = recv_buffer[i] / world_size;
    }
    
    std::cout << "DEBUG [Rank " << rank << "]: First element after sync: " << ptr[0] << std::endl;
    
    // Check if Python can actually see our changes
    std::cout << "DEBUG [Rank " << rank << "]: Checking buffer writability: " << (buf.readonly ? "READ-ONLY" : "WRITABLE") << std::endl;
    
    // Clean up
    delete[] recv_buffer;
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

PYBIND11_MODULE(mpi_wrapper, m) {
    m.doc() = "Python bindings for MPI gradient synchronization";
    
    m.def("init_mpi", &init_mpi_wrapper, "Initialize MPI");
    m.def("sync_gradients", &sync_gradients_wrapper, "Synchronize gradients across processes");
    m.def("finalize_mpi", &finalize_mpi_wrapper, "Finalize MPI");
    m.def("get_world_size", &get_world_size_wrapper, "Get the number of processes");
    m.def("get_world_rank", &get_world_rank_wrapper, "Get the current process rank");
} 