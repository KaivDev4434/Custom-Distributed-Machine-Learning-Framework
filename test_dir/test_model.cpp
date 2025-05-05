#include "../src/model/Model.h"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "Starting Model test..." << std::endl;
        
        // Create a model with the same parameters as in benchmarks
        Model model(1, 64);
        
        // Create test input (64 samples, 784 features each)
        std::vector<float> input(64 * 784, 1.0f);
        
        // Test forward pass
        std::cout << "Testing forward pass..." << std::endl;
        model.forward(input.data());
        
        // Get output
        std::cout << "Testing get_output..." << std::endl;
        const float* output = model.get_output();
        
        // Verify output
        std::vector<float> host_output(64 * 10);
        cudaMemcpy(host_output.data(), output, 64 * 10 * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Print first few output values
        std::cout << "Output values:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << host_output[i] << " ";
        }
        std::cout << std::endl;
        
        // Success!
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
    } 
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 