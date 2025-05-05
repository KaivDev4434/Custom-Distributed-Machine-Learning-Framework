#pragma once

#include <cuda_runtime.h>

// Forward declarations
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void relu_kernel(const float* input, float* output, int N);
__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, int N);
__global__ void softmax_kernel(const float* input, float* output, int batch_size, int num_classes);
__global__ void softmax_cross_entropy_kernel(const float* logits, const int* labels, float* loss, int batch_size, int num_classes);
__global__ void softmax_cross_entropy_backward_kernel(const float* probs, const int* labels, float* grad_input, int batch_size, int num_classes);
__global__ void gradient_sync_kernel(float* gradients, int size, int num_gpus);

// CUDA wrapper functions to launch kernels
cudaError_t matmul(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream = 0);
cudaError_t add_bias(float* output, const float* bias, int batch_size, int dim, cudaStream_t stream);
cudaError_t relu_forward(const float* input, float* output, int N, cudaStream_t stream = 0);
cudaError_t relu_backward(const float* grad_output, const float* input, float* grad_input, int N, cudaStream_t stream = 0);
cudaError_t softmax_forward(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream = 0);
cudaError_t softmax_cross_entropy_loss(const float* logits, const int* labels, float* loss, int batch_size, int num_classes, cudaStream_t stream = 0);
cudaError_t softmax_cross_entropy_grad(const float* probs, const int* labels, float* grad_input, int batch_size, int num_classes, cudaStream_t stream = 0);
cudaError_t synchronize_gradients(float* gradients, int size, int num_gpus, cudaStream_t stream = 0);

// Utility functions
cudaError_t cuda_check(cudaError_t result); 