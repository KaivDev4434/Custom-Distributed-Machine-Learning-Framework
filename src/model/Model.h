#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

class Model {
public:
    // Updated constructor to match Python interface
    Model(int input_size, int hidden_size, int num_classes);
    ~Model() noexcept;

    // Forward pass
    void forward(const float* input);
    
    // Backward pass
    void backward(const float* input, const int* labels);
    
    // Get output probabilities
    const float* get_output() const;
    
    // Get output size
    int get_output_size() const { return batch_size_ * num_classes_; }
    
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

    // Helper functions
    void synchronize();

private:
    // Model parameters
    int input_size_;     // 784 for MNIST
    int hidden_size_;    // 128 in our case
    int num_classes_;    // 10 for MNIST
    int batch_size_;     // 64 in our case
    
    // FC1 layer (input_size -> hidden_size)
    float* fc1_weights_device_;    // hidden_size x input_size
    float* fc1_bias_device_;       // hidden_size
    float* fc1_output_device_;     // batch_size x hidden_size
    
    // FC2 layer (hidden_size -> num_classes)
    float* fc2_weights_device_;    // num_classes x hidden_size
    float* fc2_bias_device_;       // num_classes
    float* fc2_output_device_;     // batch_size x num_classes
    
    // Gradients
    float* grad_fc1_weights_device_;
    float* grad_fc1_bias_device_;
    float* grad_fc2_weights_device_;
    float* grad_fc2_bias_device_;
    
    // Intermediate gradients
    float* grad_fc1_output_device_;
    float* grad_fc2_output_device_;
    
    // CUDA stream
    cudaStream_t stream_;
    
    // CUDA handles
    cublasHandle_t handle_;
    
    // Helper functions
    void allocate_memory();
    void free_memory();
    void initialize_weights();
    void update_weights(float learning_rate);
}; 