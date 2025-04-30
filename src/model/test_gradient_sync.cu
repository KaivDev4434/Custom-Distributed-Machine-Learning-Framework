#include "cuda_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Function to initialize gradients with different values for each GPU
void initialize_gradients(float* gradients, int size, int num_gpus) {
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        for (int i = 0; i < size; ++i) {
            gradients[gpu * size + i] = (gpu + 1) * 10.0f + i;  // Different values for each GPU
        }
    }
}

// Function to verify synchronized gradients
bool verify_sync(float* gradients, int size, int num_gpus) {
    bool success = true;
    for (int i = 0; i < size; ++i) {
        float expected = 0.0f;
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            expected += (gpu + 1) * 10.0f + i;
        }
        expected /= num_gpus;
        
        if (fabs(gradients[i] - expected) > 1e-5) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, gradients[i]);
            success = false;
        }
    }
    return success;
}

int main() {
    const int size = 1000;  // Size of gradients
    const int num_gpus = 4; // Number of GPUs to simulate
    
    // Allocate unified memory for gradients (accessible by all GPUs)
    float* gradients;
    cudaMallocManaged(&gradients, size * num_gpus * sizeof(float));
    
    // Initialize gradients
    initialize_gradients(gradients, size, num_gpus);
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Synchronize gradients
    cudaError_t error = synchronize_gradients(gradients, size, num_gpus, stream);
    if (error != cudaSuccess) {
        printf("Error during gradient synchronization: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // Wait for synchronization to complete
    cudaStreamSynchronize(stream);
    
    // Verify results
    bool success = verify_sync(gradients, size, num_gpus);
    if (success) {
        printf("Gradient synchronization test passed!\n");
    } else {
        printf("Gradient synchronization test failed!\n");
    }
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(gradients);
    
    return success ? 0 : 1;
} 