#include "Model.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// CUDA kernels
__global__ void initialize_weights_kernel(float* weights, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] = (rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void softmax_cross_entropy_kernel(
    const float* input, const int* labels, float* output, float* loss,
    int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Find max for numerical stability
        float max_val = input[idx * num_classes];
        for (int i = 1; i < num_classes; i++) {
            max_val = max(max_val, input[idx * num_classes + i]);
        }
        
        // Compute softmax
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[idx * num_classes + i] = exp(input[idx * num_classes + i] - max_val);
            sum += output[idx * num_classes + i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[idx * num_classes + i] /= sum;
        }
        
        // Compute loss
        float label_val = output[idx * num_classes + labels[idx]];
        atomicAdd(loss, -log(label_val + 1e-10f));
    }
}

__global__ void softmax_cross_entropy_backward_kernel(
    const float* output, const int* labels, float* grad_input,
    int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        for (int i = 0; i < num_classes; i++) {
            grad_input[idx * num_classes + i] = output[idx * num_classes + i];
            if (i == labels[idx]) {
                grad_input[idx * num_classes + i] -= 1.0f;
            }
        }
    }
}

Model::Model(int input_size, int hidden_size, int num_classes)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_classes_(num_classes)
    , batch_size_(64)  // Fixed batch size for now
{
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    // Allocate memory
    allocate_memory();
    
    // Initialize weights
    initialize_weights();
}

Model::~Model() {
    free_memory();
    CUDA_CHECK(cudaStreamDestroy(stream_));
}

void Model::allocate_memory() {
    // FC1 layer
    CUDA_CHECK(cudaMalloc(&fc1_weights_device_, hidden_size_ * input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc1_bias_device_, hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc1_output_device_, batch_size_ * hidden_size_ * sizeof(float)));
    
    // FC2 layer
    CUDA_CHECK(cudaMalloc(&fc2_weights_device_, num_classes_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc2_bias_device_, num_classes_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc2_output_device_, batch_size_ * num_classes_ * sizeof(float)));
    
    // Gradients
    CUDA_CHECK(cudaMalloc(&grad_fc1_weights_device_, hidden_size_ * input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc1_bias_device_, hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc2_weights_device_, num_classes_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc2_bias_device_, num_classes_ * sizeof(float)));
    
    // Intermediate gradients
    CUDA_CHECK(cudaMalloc(&grad_fc1_output_device_, batch_size_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc2_output_device_, batch_size_ * num_classes_ * sizeof(float)));
}

void Model::free_memory() {
    // FC1 layer
    CUDA_CHECK(cudaFree(fc1_weights_device_));
    CUDA_CHECK(cudaFree(fc1_bias_device_));
    CUDA_CHECK(cudaFree(fc1_output_device_));
    
    // FC2 layer
    CUDA_CHECK(cudaFree(fc2_weights_device_));
    CUDA_CHECK(cudaFree(fc2_bias_device_));
    CUDA_CHECK(cudaFree(fc2_output_device_));
    
    // Gradients
    CUDA_CHECK(cudaFree(grad_fc1_weights_device_));
    CUDA_CHECK(cudaFree(grad_fc1_bias_device_));
    CUDA_CHECK(cudaFree(grad_fc2_weights_device_));
    CUDA_CHECK(cudaFree(grad_fc2_bias_device_));
    
    // Intermediate gradients
    CUDA_CHECK(cudaFree(grad_fc1_output_device_));
    CUDA_CHECK(cudaFree(grad_fc2_output_device_));
}

void Model::initialize_weights() {
    // Initialize FC1 weights and bias
    initialize_weights_kernel<<<(hidden_size_ * input_size_ + 255) / 256, 256, 0, stream_>>>(
        fc1_weights_device_, hidden_size_ * input_size_, sqrt(2.0f / input_size_));
    initialize_weights_kernel<<<(hidden_size_ + 255) / 256, 256, 0, stream_>>>(
        fc1_bias_device_, hidden_size_, 0.0f);
    
    // Initialize FC2 weights and bias
    initialize_weights_kernel<<<(num_classes_ * hidden_size_ + 255) / 256, 256, 0, stream_>>>(
        fc2_weights_device_, num_classes_ * hidden_size_, sqrt(2.0f / hidden_size_));
    initialize_weights_kernel<<<(num_classes_ + 255) / 256, 256, 0, stream_>>>(
        fc2_bias_device_, num_classes_, 0.0f);
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Model::forward(const float* input) {
    // Create CUBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream_);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // FC1: input -> hidden
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                hidden_size_, batch_size_, input_size_,
                &alpha, fc1_weights_device_, hidden_size_,
                input, input_size_,
                &beta, fc1_output_device_, hidden_size_);
    
    // Add bias
    for (int i = 0; i < batch_size_; i++) {
        cublasSaxpy(handle, hidden_size_,
                    &alpha, fc1_bias_device_, 1,
                    fc1_output_device_ + i * hidden_size_, 1);
    }
    
    // ReLU activation
    relu_forward_kernel<<<(batch_size_ * hidden_size_ + 255) / 256, 256, 0, stream_>>>(
        fc1_output_device_, fc1_output_device_, batch_size_ * hidden_size_);
    
    // FC2: hidden -> output
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                num_classes_, batch_size_, hidden_size_,
                &alpha, fc2_weights_device_, num_classes_,
                fc1_output_device_, hidden_size_,
                &beta, fc2_output_device_, num_classes_);
    
    // Add bias
    for (int i = 0; i < batch_size_; i++) {
        cublasSaxpy(handle, num_classes_,
                    &alpha, fc2_bias_device_, 1,
                    fc2_output_device_ + i * num_classes_, 1);
    }
    
    cublasDestroy(handle);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Model::backward(const float* input, const int* labels) {
    // Create CUBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream_);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Compute softmax and cross-entropy loss
    float* loss_device;
    CUDA_CHECK(cudaMalloc(&loss_device, sizeof(float)));
    CUDA_CHECK(cudaMemset(loss_device, 0, sizeof(float)));
    
    softmax_cross_entropy_kernel<<<(batch_size_ + 255) / 256, 256, 0, stream_>>>(
        fc2_output_device_, labels, fc2_output_device_, loss_device,
        batch_size_, num_classes_);
    
    // Backpropagate through softmax and cross-entropy
    softmax_cross_entropy_backward_kernel<<<(batch_size_ + 255) / 256, 256, 0, stream_>>>(
        fc2_output_device_, labels, grad_fc2_output_device_,
        batch_size_, num_classes_);
    
    // Backpropagate through FC2
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                hidden_size_, num_classes_, batch_size_,
                &alpha, fc1_output_device_, hidden_size_,
                grad_fc2_output_device_, num_classes_,
                &beta, grad_fc2_weights_device_, hidden_size_);
    
    // Compute bias gradients
    for (int i = 0; i < batch_size_; i++) {
        cublasSaxpy(handle, num_classes_,
                    &alpha, grad_fc2_output_device_ + i * num_classes_, 1,
                    grad_fc2_bias_device_, 1);
    }
    
    // Backpropagate through ReLU
    relu_backward_kernel<<<(batch_size_ * hidden_size_ + 255) / 256, 256, 0, stream_>>>(
        grad_fc2_output_device_, fc1_output_device_, grad_fc1_output_device_,
        batch_size_ * hidden_size_);
    
    // Backpropagate through FC1
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                input_size_, hidden_size_, batch_size_,
                &alpha, input, input_size_,
                grad_fc1_output_device_, hidden_size_,
                &beta, grad_fc1_weights_device_, input_size_);
    
    // Compute bias gradients
    for (int i = 0; i < batch_size_; i++) {
        cublasSaxpy(handle, hidden_size_,
                    &alpha, grad_fc1_output_device_ + i * hidden_size_, 1,
                    grad_fc1_bias_device_, 1);
    }
    
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(loss_device));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

const float* Model::get_output() const {
    return fc2_output_device_;
}

float Model::get_loss() const {
    float* loss_host = new float[1];
    float* loss_device;
    CUDA_CHECK(cudaMalloc(&loss_device, sizeof(float)));
    CUDA_CHECK(cudaMemset(loss_device, 0, sizeof(float)));
    
    // Compute softmax and cross-entropy loss
    softmax_cross_entropy_kernel<<<(batch_size_ + 255) / 256, 256, 0, stream_>>>(
        fc2_output_device_, nullptr, fc2_output_device_, loss_device,
        batch_size_, num_classes_);
    
    CUDA_CHECK(cudaMemcpyAsync(loss_host, loss_device, sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    float loss = loss_host[0];
    delete[] loss_host;
    CUDA_CHECK(cudaFree(loss_device));
    return loss;
}

void Model::update_weights(float learning_rate) {
    const float alpha = -learning_rate;
    
    // Update FC1 weights and bias
    cublasSaxpy(handle_, hidden_size_ * input_size_,
                &alpha, grad_fc1_weights_device_, 1,
                fc1_weights_device_, 1);
    cublasSaxpy(handle_, hidden_size_,
                &alpha, grad_fc1_bias_device_, 1,
                fc1_bias_device_, 1);
    
    // Update FC2 weights and bias
    cublasSaxpy(handle_, num_classes_ * hidden_size_,
                &alpha, grad_fc2_weights_device_, 1,
                fc2_weights_device_, 1);
    cublasSaxpy(handle_, num_classes_,
                &alpha, grad_fc2_bias_device_, 1,
                fc2_bias_device_, 1);
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Model::update(float learning_rate) {
    update_weights(learning_rate);
}

void Model::zero_grad() {
    // Zero FC1 gradients
    CUDA_CHECK(cudaMemset(grad_fc1_weights_device_, 0, hidden_size_ * input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_fc1_bias_device_, 0, hidden_size_ * sizeof(float)));
    
    // Zero FC2 gradients
    CUDA_CHECK(cudaMemset(grad_fc2_weights_device_, 0, num_classes_ * hidden_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_fc2_bias_device_, 0, num_classes_ * sizeof(float)));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Model::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Model::save(const std::string& filename) const {
    // Not implemented yet
}

void Model::load(const std::string& filename) {
    // Not implemented yet
} 