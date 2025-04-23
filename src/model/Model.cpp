#include "Model.h"
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>

Model::Model(int input_size, int hidden_size, int num_classes)
    : input_size_(input_size), hidden_size_(hidden_size), num_classes_(num_classes),
      W1_host_(input_size * hidden_size), b1_host_(hidden_size),
      W2_host_(hidden_size * num_classes), b2_host_(num_classes),
      dW1_host_(input_size * hidden_size), db1_host_(hidden_size),
      dW2_host_(hidden_size * num_classes), db2_host_(num_classes),
      loss_host_(0.0f) {
          
    // Initialize weights
    initialize_weights();
    
    // Allocate device memory
    allocate_memory();
    
    // Copy weights to device
    copy_to_device();
    
    // Create CUDA stream
    cuda_check(cudaStreamCreate(&stream_));
}

Model::~Model() {
    // Free device memory
    free_memory();
    
    // Destroy CUDA stream
    cudaStreamDestroy(stream_);
}

void Model::forward(const float* input, int batch_size) {
    // Device memory for input
    float* input_device;
    cuda_check(cudaMalloc(&input_device, sizeof(float) * batch_size * input_size_));
    cuda_check(cudaMemcpyAsync(input_device, input, sizeof(float) * batch_size * input_size_, 
                              cudaMemcpyHostToDevice, stream_));
    
    // Hidden layer: hidden_input = input * W1 + b1
    cuda_check(matmul(input_device, W1_device_, hidden_input_device_, batch_size, hidden_size_, input_size_, stream_));
    
    // Add bias (TODO: Implement bias addition kernel)
    // For simplicity, we'll skip bias addition for now
    
    // ReLU activation: hidden_output = ReLU(hidden_input)
    cuda_check(relu_forward(hidden_input_device_, hidden_output_device_, batch_size * hidden_size_, stream_));
    
    // Output layer: output_logits = hidden_output * W2 + b2
    cuda_check(matmul(hidden_output_device_, W2_device_, output_logits_device_, batch_size, num_classes_, hidden_size_, stream_));
    
    // Add bias (TODO: Implement bias addition kernel)
    // For simplicity, we'll skip bias addition for now
    
    // Softmax: output_probs = softmax(output_logits)
    cuda_check(softmax_forward(output_logits_device_, output_probs_device_, batch_size, num_classes_, stream_));
    
    // Free input memory
    cudaFree(input_device);
}

void Model::backward(const int* labels, int batch_size) {
    // Device memory for labels
    int* labels_device;
    cuda_check(cudaMalloc(&labels_device, sizeof(int) * batch_size));
    cuda_check(cudaMemcpyAsync(labels_device, labels, sizeof(int) * batch_size, 
                              cudaMemcpyHostToDevice, stream_));
    
    // Compute loss and gradient
    cuda_check(softmax_cross_entropy_loss(output_logits_device_, labels_device, loss_device_, batch_size, num_classes_, stream_));
    
    // Copy loss to host
    cuda_check(cudaMemcpyAsync(&loss_host_, loss_device_, sizeof(float), cudaMemcpyDeviceToHost, stream_));
    
    // Compute output gradient: grad_output = softmax(output_logits) - one_hot(labels)
    cuda_check(softmax_cross_entropy_grad(output_probs_device_, labels_device, grad_output_device_, batch_size, num_classes_, stream_));
    
    // Backprop to hidden layer: grad_hidden_output = grad_output * W2^T
    cuda_check(matmul(grad_output_device_, W2_device_, grad_hidden_output_device_, batch_size, hidden_size_, num_classes_, stream_));
    
    // ReLU backward: grad_hidden_input = grad_hidden_output * (hidden_input > 0)
    cuda_check(relu_backward(grad_hidden_output_device_, hidden_input_device_, grad_hidden_input_device_, batch_size * hidden_size_, stream_));
    
    // Compute weight gradients
    // dW2 = hidden_output^T * grad_output
    // For simplicity, we'll use a simplified version where we don't transpose hidden_output
    cuda_check(matmul(hidden_output_device_, grad_output_device_, dW2_device_, hidden_size_, num_classes_, batch_size, stream_));
    
    // db2 = sum(grad_output, dim=0)
    // TODO: Implement reduction kernel for summing over batch dimension
    
    // dW1 = input^T * grad_hidden_input
    // TODO: Implement matrix transpose and multiplication for input and grad_hidden_input
    
    // db1 = sum(grad_hidden_input, dim=0)
    // TODO: Implement reduction kernel for summing over batch dimension
    
    // Sync before returning
    cuda_check(cudaStreamSynchronize(stream_));
    
    // Free label memory
    cudaFree(labels_device);
}

const float* Model::get_output() const {
    return output_probs_device_;
}

float Model::get_loss() const {
    return loss_host_;
}

void Model::update(float learning_rate) {
    // TODO: Implement SGD update kernel
    // For now, we'll copy gradients to host, update, and copy back
    
    // Copy gradients to host
    cuda_check(cudaMemcpyAsync(dW2_host_.data(), dW2_device_, sizeof(float) * hidden_size_ * num_classes_, 
                              cudaMemcpyDeviceToHost, stream_));
    
    // Update weights on host
    for (int i = 0; i < hidden_size_ * num_classes_; ++i) {
        W2_host_[i] -= learning_rate * dW2_host_[i];
    }
    
    // Copy updated weights back to device
    cuda_check(cudaMemcpyAsync(W2_device_, W2_host_.data(), sizeof(float) * hidden_size_ * num_classes_, 
                              cudaMemcpyHostToDevice, stream_));
    
    // Sync before returning
    cuda_check(cudaStreamSynchronize(stream_));
}

void Model::zero_grad() {
    // Reset gradients to zero
    cuda_check(cudaMemsetAsync(dW1_device_, 0, sizeof(float) * input_size_ * hidden_size_, stream_));
    cuda_check(cudaMemsetAsync(db1_device_, 0, sizeof(float) * hidden_size_, stream_));
    cuda_check(cudaMemsetAsync(dW2_device_, 0, sizeof(float) * hidden_size_ * num_classes_, stream_));
    cuda_check(cudaMemsetAsync(db2_device_, 0, sizeof(float) * num_classes_, stream_));
}

void Model::save(const std::string& filename) const {
    // Copy weights from device to host
    std::vector<float> W1(W1_host_.size());
    std::vector<float> b1(b1_host_.size());
    std::vector<float> W2(W2_host_.size());
    std::vector<float> b2(b2_host_.size());
    
    cudaMemcpy(W1.data(), W1_device_, sizeof(float) * W1.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1.data(), b1_device_, sizeof(float) * b1.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(W2.data(), W2_device_, sizeof(float) * W2.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2.data(), b2_device_, sizeof(float) * b2.size(), cudaMemcpyDeviceToHost);
    
    // Save to file
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for saving: " << filename << std::endl;
        return;
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&input_size_), sizeof(input_size_));
    file.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(hidden_size_));
    file.write(reinterpret_cast<const char*>(&num_classes_), sizeof(num_classes_));
    
    // Write weights
    file.write(reinterpret_cast<const char*>(W1.data()), sizeof(float) * W1.size());
    file.write(reinterpret_cast<const char*>(b1.data()), sizeof(float) * b1.size());
    file.write(reinterpret_cast<const char*>(W2.data()), sizeof(float) * W2.size());
    file.write(reinterpret_cast<const char*>(b2.data()), sizeof(float) * b2.size());
    
    file.close();
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for loading: " << filename << std::endl;
        return;
    }
    
    // Read dimensions
    int input_size, hidden_size, num_classes;
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    file.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));
    
    // Check if dimensions match
    if (input_size != input_size_ || hidden_size != hidden_size_ || num_classes != num_classes_) {
        std::cerr << "Model dimensions mismatch!" << std::endl;
        file.close();
        return;
    }
    
    // Read weights
    file.read(reinterpret_cast<char*>(W1_host_.data()), sizeof(float) * W1_host_.size());
    file.read(reinterpret_cast<char*>(b1_host_.data()), sizeof(float) * b1_host_.size());
    file.read(reinterpret_cast<char*>(W2_host_.data()), sizeof(float) * W2_host_.size());
    file.read(reinterpret_cast<char*>(b2_host_.data()), sizeof(float) * b2_host_.size());
    
    file.close();
    
    // Copy weights to device
    copy_to_device();
}

void Model::allocate_memory() {
    // Allocate memory for weights and biases
    cuda_check(cudaMalloc(&W1_device_, sizeof(float) * input_size_ * hidden_size_));
    cuda_check(cudaMalloc(&b1_device_, sizeof(float) * hidden_size_));
    cuda_check(cudaMalloc(&W2_device_, sizeof(float) * hidden_size_ * num_classes_));
    cuda_check(cudaMalloc(&b2_device_, sizeof(float) * num_classes_));
    
    // Allocate memory for gradients
    cuda_check(cudaMalloc(&dW1_device_, sizeof(float) * input_size_ * hidden_size_));
    cuda_check(cudaMalloc(&db1_device_, sizeof(float) * hidden_size_));
    cuda_check(cudaMalloc(&dW2_device_, sizeof(float) * hidden_size_ * num_classes_));
    cuda_check(cudaMalloc(&db2_device_, sizeof(float) * num_classes_));
    
    // Allocate memory for activations
    cuda_check(cudaMalloc(&hidden_input_device_, sizeof(float) * 100 * hidden_size_));  // Assuming max batch size of 100
    cuda_check(cudaMalloc(&hidden_output_device_, sizeof(float) * 100 * hidden_size_));
    cuda_check(cudaMalloc(&output_logits_device_, sizeof(float) * 100 * num_classes_));
    cuda_check(cudaMalloc(&output_probs_device_, sizeof(float) * 100 * num_classes_));
    
    // Allocate memory for gradients
    cuda_check(cudaMalloc(&grad_output_device_, sizeof(float) * 100 * num_classes_));
    cuda_check(cudaMalloc(&grad_hidden_output_device_, sizeof(float) * 100 * hidden_size_));
    cuda_check(cudaMalloc(&grad_hidden_input_device_, sizeof(float) * 100 * hidden_size_));
    
    // Allocate memory for loss
    cuda_check(cudaMalloc(&loss_device_, sizeof(float)));
}

void Model::free_memory() {
    // Free memory for weights and biases
    cudaFree(W1_device_);
    cudaFree(b1_device_);
    cudaFree(W2_device_);
    cudaFree(b2_device_);
    
    // Free memory for gradients
    cudaFree(dW1_device_);
    cudaFree(db1_device_);
    cudaFree(dW2_device_);
    cudaFree(db2_device_);
    
    // Free memory for activations
    cudaFree(hidden_input_device_);
    cudaFree(hidden_output_device_);
    cudaFree(output_logits_device_);
    cudaFree(output_probs_device_);
    
    // Free memory for gradients
    cudaFree(grad_output_device_);
    cudaFree(grad_hidden_output_device_);
    cudaFree(grad_hidden_input_device_);
    
    // Free memory for loss
    cudaFree(loss_device_);
}

void Model::copy_to_device() {
    cuda_check(cudaMemcpy(W1_device_, W1_host_.data(), sizeof(float) * W1_host_.size(), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b1_device_, b1_host_.data(), sizeof(float) * b1_host_.size(), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(W2_device_, W2_host_.data(), sizeof(float) * W2_host_.size(), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(b2_device_, b2_host_.data(), sizeof(float) * b2_host_.size(), cudaMemcpyHostToDevice));
}

void Model::copy_from_device() {
    cuda_check(cudaMemcpy(W1_host_.data(), W1_device_, sizeof(float) * W1_host_.size(), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b1_host_.data(), b1_device_, sizeof(float) * b1_host_.size(), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(W2_host_.data(), W2_device_, sizeof(float) * W2_host_.size(), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b2_host_.data(), b2_device_, sizeof(float) * b2_host_.size(), cudaMemcpyDeviceToHost));
}

void Model::initialize_weights() {
    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // For W1
    float stddev = std::sqrt(2.0f / (input_size_ + hidden_size_));
    std::normal_distribution<float> dist1(0.0f, stddev);
    for (auto& w : W1_host_) {
        w = dist1(gen);
    }
    
    // For W2
    stddev = std::sqrt(2.0f / (hidden_size_ + num_classes_));
    std::normal_distribution<float> dist2(0.0f, stddev);
    for (auto& w : W2_host_) {
        w = dist2(gen);
    }
    
    // Initialize biases to zero
    std::fill(b1_host_.begin(), b1_host_.end(), 0.0f);
    std::fill(b2_host_.begin(), b2_host_.end(), 0.0f);
    
    // Initialize gradients to zero
    std::fill(dW1_host_.begin(), dW1_host_.end(), 0.0f);
    std::fill(db1_host_.begin(), db1_host_.end(), 0.0f);
    std::fill(dW2_host_.begin(), dW2_host_.end(), 0.0f);
    std::fill(db2_host_.begin(), db2_host_.end(), 0.0f);
} 