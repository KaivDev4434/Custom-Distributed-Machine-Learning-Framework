#include "cuda_kernels.cuh"
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256
#define TILE_WIDTH 16

// Utility function to check CUDA errors
cudaError_t cuda_check(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}

// Bias addition kernel
__global__ void add_bias_kernel(float* output, const float* bias, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / dim;
    int dim_idx = idx % dim;
    
    if (batch_idx < batch_size) {
        output[batch_idx * dim + dim_idx] += bias[dim_idx];
    }
}

// Matrix multiplication kernel (C = A * B)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory for tile
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within tile
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Global row and column
    int globalRow = blockRow * TILE_WIDTH + row;
    int globalCol = blockCol * TILE_WIDTH + col;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load tile to shared memory
        if (globalRow < M && t * TILE_WIDTH + col < K) {
            As[row][col] = A[globalRow * K + t * TILE_WIDTH + col];
        } else {
            As[row][col] = 0.0f;
        }
        
        if (t * TILE_WIDTH + row < K && globalCol < N) {
            Bs[row][col] = B[(t * TILE_WIDTH + row) * N + globalCol];
        } else {
            Bs[row][col] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[row][k] * Bs[k][col];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = sum;
    }
}

// ReLU activation forward
__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU backward pass
__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
    }
}

// Softmax forward pass
__global__ void softmax_kernel(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; ++i) {
            max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            float exp_val = expf(input[batch_idx * num_classes + i] - max_val);
            output[batch_idx * num_classes + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < num_classes; ++i) {
            output[batch_idx * num_classes + i] /= sum;
        }
    }
}

// Softmax cross entropy loss forward
__global__ void softmax_cross_entropy_kernel(const float* logits, const int* labels, float* loss, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; ++i) {
            max_val = fmaxf(max_val, logits[batch_idx * num_classes + i]);
        }
        
        // Compute log-sum-exp
        float log_sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            log_sum_exp += expf(logits[batch_idx * num_classes + i] - max_val);
        }
        log_sum_exp = logf(log_sum_exp) + max_val;
        
        // Compute loss for this sample
        int label = labels[batch_idx];
        float sample_loss = log_sum_exp - logits[batch_idx * num_classes + label];
        
        // Atomic add to total loss
        atomicAdd(loss, sample_loss / batch_size);
    }
}

// Softmax cross entropy backward
__global__ void softmax_cross_entropy_backward_kernel(const float* probs, const int* labels, float* grad_input, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / num_classes;
    int class_idx = idx % num_classes;
    
    if (batch_idx < batch_size) {
        int label = labels[batch_idx];
        grad_input[idx] = probs[idx] - (class_idx == label ? 1.0f : 0.0f);
    }
}

// Gradient synchronization kernel
__global__ void gradient_sync_kernel(float* gradients, int size, int num_gpus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Sum gradients from all GPUs
        float sum = 0.0f;
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            sum += gradients[gpu * size + idx];
        }
        // Average the gradients
        gradients[idx] = sum / num_gpus;
    }
}

// Wrapper functions to launch kernels

cudaError_t add_bias(float* output, const float* bias, int batch_size, int dim, cudaStream_t stream) {
    int blockSize = BLOCK_SIZE;
    int numBlocks = (batch_size * dim + blockSize - 1) / blockSize;
    
    add_bias_kernel<<<numBlocks, blockSize, 0, stream>>>(output, bias, batch_size, dim);
    return cudaGetLastError();
}

cudaError_t matmul(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    matmul_kernel<<<dimGrid, dimBlock, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

cudaError_t relu_forward(const float* input, float* output, int N, cudaStream_t stream) {
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    relu_kernel<<<numBlocks, blockSize, 0, stream>>>(input, output, N);
    return cudaGetLastError();
}

cudaError_t relu_backward(const float* grad_output, const float* input, float* grad_input, int N, cudaStream_t stream) {
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    relu_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_output, input, grad_input, N);
    return cudaGetLastError();
}

cudaError_t softmax_forward(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream) {
    softmax_kernel<<<batch_size, 1, 0, stream>>>(input, output, batch_size, num_classes);
    return cudaGetLastError();
}

cudaError_t softmax_cross_entropy_loss(const float* logits, const int* labels, float* loss, int batch_size, int num_classes, cudaStream_t stream) {
    // Reset loss to zero
    cudaMemsetAsync(loss, 0, sizeof(float), stream);
    
    int blockSize = BLOCK_SIZE;
    int numBlocks = (batch_size + blockSize - 1) / blockSize;
    
    softmax_cross_entropy_kernel<<<numBlocks, blockSize, 0, stream>>>(logits, labels, loss, batch_size, num_classes);
    return cudaGetLastError();
}

cudaError_t softmax_cross_entropy_grad(const float* probs, const int* labels, float* grad_input, int batch_size, int num_classes, cudaStream_t stream) {
    int blockSize = BLOCK_SIZE;
    int numBlocks = (batch_size * num_classes + blockSize - 1) / blockSize;
    
    softmax_cross_entropy_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(probs, labels, grad_input, batch_size, num_classes);
    return cudaGetLastError();
}

// Wrapper function for gradient synchronization
cudaError_t synchronize_gradients(float* gradients, int size, int num_gpus, cudaStream_t stream) {
    int blockSize = BLOCK_SIZE;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    gradient_sync_kernel<<<numBlocks, blockSize, 0, stream>>>(gradients, size, num_gpus);
    return cudaGetLastError();
} 