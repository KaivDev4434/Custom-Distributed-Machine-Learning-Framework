#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "lib"))
import nn_model as model_module

from torchvision import datasets, transforms

# Ensure CUDA compatibility
CUDA_VERSION = torch.version.cuda
print(f"Using CUDA version: {CUDA_VERSION}")

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 13x13 -> 11x11
        # After pooling twice: 5x5 with 64 channels = 1600
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2) # 26x26 -> 13x13
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2) # 11x11 -> 5x5
        x = torch.flatten(x, 1) # Flatten: 64 channels * 5*5 = 1600
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def benchmark_custom_model():
    # Create model with correct dimensions for MNIST
    # Input size: 28*28 = 784, Hidden size: 128, Output classes: 10
    model = model_module.Model(784, 128, 10)
    
    # Create dummy input - must be flattened for the model and converted to numpy
    input_tensor = torch.randn(64, 784).cuda()
    input_data = input_tensor.cpu().numpy().astype(np.float32)
    
    # Check CUDA device
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    print(f"Using GPU: {device_name}")
    
    # Warm up
    for _ in range(10):
        _ = model.forward(input_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = model.forward(input_data)
    torch.cuda.synchronize()  # Ensure all CUDA operations are completed
    end_time = time.time()
    return end_time - start_time

def benchmark_pytorch_model():
    model = PyTorchModel().cuda()
    # Create 4D input for CNN (batch_size, channels, height, width)
    input_data = torch.randn(64, 1, 28, 28).cuda()
    
    # Warm up
    for _ in range(10):
        _ = model(input_data)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = model(input_data)
    torch.cuda.synchronize()  # Ensure all CUDA operations are completed
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    print("Benchmarking Model Stage")
    print("-----------------------")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation.")
        exit(1)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    custom_time = benchmark_custom_model()
    torch_time = benchmark_pytorch_model()
    
    print(f"\nResults:")
    print(f"Custom Model time: {custom_time:.4f} seconds")
    print(f"PyTorch Model time: {torch_time:.4f} seconds")
    print(f"Speedup: {torch_time/custom_time:.2f}x") 