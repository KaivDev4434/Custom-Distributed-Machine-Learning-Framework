#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Model.h"
#include <iostream>
#include <cuda_runtime.h>

namespace py = pybind11;

class ModelWrapper {
private:
    std::unique_ptr<Model> model_;
    float* input_device_ = nullptr;
    int* labels_device_ = nullptr;
    int input_size_ = 0;
    int batch_size_ = 64;

public:
    ModelWrapper(int input_channels, int batch_size) 
        : model_(std::make_unique<Model>(784, 128, 10)),  // Fixed dimensions for MNIST
          input_size_(784),
          batch_size_(batch_size) {
        
        // Allocate device memory for inputs and labels
        cudaMalloc(&input_device_, batch_size_ * input_size_ * sizeof(float));
        cudaMalloc(&labels_device_, batch_size_ * sizeof(int));
        
        // Initialize memory to prevent uninitialized memory issues
        cudaMemset(input_device_, 0, batch_size_ * input_size_ * sizeof(float));
        cudaMemset(labels_device_, 0, batch_size_ * sizeof(int));
    }
    
    ~ModelWrapper() {
        if (input_device_) cudaFree(input_device_);
        if (labels_device_) cudaFree(labels_device_);
    }

    void forward(py::array_t<float> input) {
        auto input_buffer = input.request();
        float* input_ptr = static_cast<float*>(input_buffer.ptr);
        
        // Copy input data to device
        cudaMemcpy(input_device_, input_ptr, batch_size_ * input_size_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        // Forward pass
        model_->forward(input_device_);
    }

    void backward(py::array_t<float> input, py::array_t<int> labels) {
        auto input_buffer = input.request();
        auto labels_buffer = labels.request();
        float* input_ptr = static_cast<float*>(input_buffer.ptr);
        int* labels_ptr = static_cast<int*>(labels_buffer.ptr);
        
        // Copy input and labels to device
        cudaMemcpy(input_device_, input_ptr, batch_size_ * input_size_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(labels_device_, labels_ptr, batch_size_ * sizeof(int), 
                   cudaMemcpyHostToDevice);
        
        // Backward pass
        model_->backward(input_device_, labels_device_);
    }

    py::array_t<float> get_output() {
        int output_size = model_->get_output_size();
        auto result = py::array_t<float>(output_size);
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        // Copy from device to host
        const float* device_output = model_->get_output();
        cudaMemcpy(ptr, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Reshape to [batch_size, num_classes]
        result.resize({batch_size_, 10});
        
        return result;
    }

    float get_loss() {
        return model_->get_loss();
    }

    void update(float learning_rate) {
        model_->update(learning_rate);
    }

    void zero_grad() {
        model_->zero_grad();
    }
};

PYBIND11_MODULE(nn_model, m) {
    m.doc() = "CUDA Model for Neural Network Training"; 
    
    py::class_<ModelWrapper>(m, "Model")
        .def(py::init<int, int>())
        .def("forward", &ModelWrapper::forward)
        .def("backward", &ModelWrapper::backward)
        .def("get_output", &ModelWrapper::get_output)
        .def("get_loss", &ModelWrapper::get_loss)
        .def("update", &ModelWrapper::update)
        .def("zero_grad", &ModelWrapper::zero_grad);
}