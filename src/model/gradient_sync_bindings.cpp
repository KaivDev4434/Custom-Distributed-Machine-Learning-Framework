#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;

// Function declarations from cuda_kernels.cuh
extern "C" {
    cudaError_t synchronize_gradients(float* gradients, int size, int num_gpus, cudaStream_t stream);
}

class GradientSync {
public:
    GradientSync() {
        // Create CUDA stream for operations
        cudaStreamCreate(&stream_);
    }
    
    ~GradientSync() {
        // Clean up CUDA stream
        cudaStreamDestroy(stream_);
    }
    
    py::array_t<float> sync_gradients(py::array_t<float> gradients, int num_gpus = 2) {
        // Get buffer info
        py::buffer_info buf = gradients.request();
        float* ptr = static_cast<float*>(buf.ptr);
        int size = buf.size / num_gpus;  // Total size divided by number of GPUs
        
        // Allocate device memory
        float* d_gradients;
        cudaMalloc(&d_gradients, buf.size * sizeof(float));
        
        // Copy data to device
        cudaMemcpy(d_gradients, ptr, buf.size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Synchronize gradients
        synchronize_gradients(d_gradients, size, num_gpus, stream_);
        
        // Wait for completion
        cudaStreamSynchronize(stream_);
        
        // Copy result back to host
        cudaMemcpy(ptr, d_gradients, buf.size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_gradients);
        
        // Return the modified input array
        return gradients;
    }
    
    float get_sync_time() {
        // Placeholder for timing functionality
        return 0.0f;
    }
    
private:
    cudaStream_t stream_;
};

PYBIND11_MODULE(gradient_sync, m) {
    m.doc() = "Python bindings for CUDA gradient synchronization";
    
    py::class_<GradientSync>(m, "GradientSync")
        .def(py::init<>())
        .def("sync_gradients", &GradientSync::sync_gradients, 
             py::arg("gradients"), py::arg("num_gpus") = 2,
             "Synchronize gradients across simulated GPUs")
        .def("get_sync_time", &GradientSync::get_sync_time, 
             "Get the time spent in synchronization");
} 