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
        // Find max logit for numerical stability
        float max_logit = -INFINITY;
        for (int c = 0; c < num_classes; ++c) {
            max_logit = fmaxf(max_logit, logits[batch_idx * num_classes + c]);
        }
        
        // Compute softmax and cross entropy loss
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += expf(logits[batch_idx * num_classes + c] - max_logit);
        }
        
        int label = labels[batch_idx];
        float log_softmax = (logits[batch_idx * num_classes + label] - max_logit) - logf(sum_exp);
        
        // Compute loss
        float batch_loss = -log_softmax;
        
        // Use atomicAdd because multiple threads might update the same loss value
        atomicAdd(loss, batch_loss / static_cast<float>(batch_size));
    }
}

// Softmax cross entropy backward
__global__ void softmax_cross_entropy_backward_kernel(const float* probs, const int* labels, float* grad_input, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / num_classes;
    int class_idx = idx % num_classes;
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        int label = labels[batch_idx];
        float grad = probs[batch_idx * num_classes + class_idx];
        
        if (class_idx == label) {
            grad -= 1.0f;
        }
        
        grad_input[batch_idx * num_classes + class_idx] = grad / static_cast<float>(batch_size);
    }
}

// Wrapper functions to launch kernels

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