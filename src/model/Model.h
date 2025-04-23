#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>

class Model {
public:
    Model(int input_size, int hidden_size, int num_classes);
    ~Model();

    // Forward pass
    void forward(const float* input, int batch_size);
    
    // Backward pass
    void backward(const int* labels, int batch_size);
    
    // Get output probabilities
    const float* get_output() const;
    
    // Get loss value
    float get_loss() const;
    
    // Update weights using SGD
    void update(float learning_rate);
    
    // Reset gradients to zero
    void zero_grad();
    
    // Save model parameters to file
    void save(const std::string& filename) const;
    
    // Load model parameters from file
    void load(const std::string& filename);

private:
    // Dimensions
    int input_size_;
    int hidden_size_;
    int num_classes_;
    
    // Weights and biases (host memory)
    std::vector<float> W1_host_; // Input -> Hidden weights
    std::vector<float> b1_host_; // Hidden bias
    std::vector<float> W2_host_; // Hidden -> Output weights
    std::vector<float> b2_host_; // Output bias
    
    // Weight gradients (host memory)
    std::vector<float> dW1_host_;
    std::vector<float> db1_host_;
    std::vector<float> dW2_host_;
    std::vector<float> db2_host_;
    
    // Device memory pointers
    float* W1_device_;
    float* b1_device_;
    float* W2_device_;
    float* b2_device_;
    
    float* dW1_device_;
    float* db1_device_;
    float* dW2_device_;
    float* db2_device_;
    
    // Activations (device memory)
    float* hidden_input_device_;  // Input to hidden layer
    float* hidden_output_device_; // Output of hidden layer after ReLU
    float* output_logits_device_; // Raw logits before softmax
    float* output_probs_device_;  // Output probabilities after softmax
    
    // Gradients (device memory)
    float* grad_output_device_;   // Gradient of loss w.r.t. output logits
    float* grad_hidden_output_device_; // Gradient of loss w.r.t. hidden output
    float* grad_hidden_input_device_;  // Gradient of loss w.r.t. hidden input
    
    // Loss (device memory)
    float* loss_device_;
    
    // Loss value (host memory)
    float loss_host_;
    
    // CUDA stream
    cudaStream_t stream_;
    
    // Helper functions
    void allocate_memory();
    void free_memory();
    void copy_to_device();
    void copy_from_device();
    void initialize_weights();
}; 