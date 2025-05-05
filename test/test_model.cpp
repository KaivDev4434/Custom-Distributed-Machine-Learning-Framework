#include "../src/model/Model.h"
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

int main() {
    try {
        // Configuration
        const int input_size = 784;  // MNIST input size
        const int hidden_size = 128;
        const int num_classes = 10;
        const int batch_size = 64;
        
        // Create model
        Model model(input_size, hidden_size, num_classes);
        
        // Generate random input data on host
        std::vector<float> host_input(batch_size * input_size);
        std::vector<int> host_labels(batch_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> label_dist(0, num_classes - 1);
        
        for (int i = 0; i < batch_size * input_size; ++i) {
            host_input[i] = dist(gen);
        }
        for (int i = 0; i < batch_size; ++i) {
            host_labels[i] = label_dist(gen);
        }
        
        // Create device memory for input and labels
        float* device_input = nullptr;
        int* device_labels = nullptr;
        
        cudaMalloc(&device_input, batch_size * input_size * sizeof(float));
        cudaMalloc(&device_labels, batch_size * sizeof(int));
        
        // Copy data to device
        cudaMemcpy(device_input, host_input.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_labels, host_labels.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
        
        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        model.forward(device_input);
        std::cout << "Forward pass completed successfully" << std::endl;
        
        // Get output size
        int output_size = model.get_output_size();
        std::cout << "Output size: " << output_size << std::endl;
        
        // Get output
        const float* device_output = model.get_output();
        std::vector<float> host_output(output_size);
        cudaMemcpy(host_output.data(), device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Backward pass
        std::cout << "Running backward pass..." << std::endl;
        model.backward(device_input, device_labels);
        std::cout << "Backward pass completed successfully" << std::endl;
        
        // Get loss
        float loss = model.get_loss();
        std::cout << "Loss: " << loss << std::endl;
        
        // Update weights
        std::cout << "Updating weights..." << std::endl;
        model.update(0.01f);
        std::cout << "Weight update completed successfully" << std::endl;
        
        // Zero gradients
        model.zero_grad();
        
        // Clean up
        cudaFree(device_input);
        cudaFree(device_labels);
        
        std::cout << "All tests passed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}