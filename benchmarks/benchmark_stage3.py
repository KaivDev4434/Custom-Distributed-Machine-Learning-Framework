#!/usr/bin/env python3
import time
import torch
import numpy as np
import sys
import os
import argparse

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "lib"))

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Stage 3: Gradient Synchronization')
    parser.add_argument('--num-gpus', type=int, default=2, 
                       help='Number of GPUs to simulate for gradient sync')
    parser.add_argument('--gradient-size', type=int, default=1000000,
                       help='Size of gradient tensors')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of sync iterations to benchmark')
    return parser.parse_args()

# Create a stub/placeholder for gradient sync
class PlaceholderGradientSync:
    def __init__(self):
        self.sync_time = 0.0
        
    def sync_gradients(self, gradients):
        start_time = time.time()
        # Simulate gradient synchronization
        time.sleep(0.01)  # Small sleep to simulate some work
        self.sync_time += time.time() - start_time
        return gradients
    
    def get_sync_time(self):
        return self.sync_time

# Try to import the real GradientSync, fall back to placeholder
try:
    # For CUDA-based gradient sync (not MPI)
    import gradient_sync
    GradientSync = gradient_sync.GradientSync
    print("Successfully imported gradient_sync module")
except ImportError as e:
    print(f"Note: Using placeholder GradientSync implementation. Error: {e}")
    GradientSync = PlaceholderGradientSync

def benchmark_custom_cuda_sync(num_gpus=2, gradient_size=1000000, iterations=50):
    # Create dummy gradients (simulate multiple GPUs)
    if torch.cuda.is_available():
        # Create a tensor to simulate multiple GPU gradients in a single tensor
        # For N GPUs, we'll create a tensor of size N*gradient_size
        gradients = torch.randn(num_gpus * gradient_size, dtype=torch.float32).cuda()
        print(f"Created CUDA tensor of shape {gradients.shape}")
    else:
        gradients = torch.randn(num_gpus * gradient_size, dtype=torch.float32)
        print("CUDA not available, using CPU tensor")
    
    # Convert to numpy array for the C++ module
    gradients_np = gradients.cpu().numpy().astype(np.float32)
    
    # Initialize custom gradient sync
    try:
        sync = GradientSync()
        print("Created GradientSync instance")
    except Exception as e:
        print(f"Error creating GradientSync: {e}")
        return 0.0
    
    # Warm up
    print("Warming up...")
    try:
        for _ in range(5):
            result = sync.sync_gradients(gradients_np)
        print("Warm-up completed successfully")
    except Exception as e:
        print(f"Error during warm-up: {e}")
        return 0.0
    
    # Benchmark
    print(f"Running benchmark with {iterations} iterations...")
    start_time = time.time()
    try:
        for _ in range(iterations):
            result = sync.sync_gradients(gradients_np)
        print("Benchmark completed successfully")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 0.0
    
    end_time = time.time()
    
    # Copy result back to CUDA if needed
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Make sure CUDA is done
    
    return end_time - start_time

def benchmark_pytorch_ddp():
    # This is a placeholder for PyTorch DDP benchmark
    # In reality, you would need to set up a proper DDP environment
    return 0.0

if __name__ == "__main__":
    args = parse_args()
    
    print("Benchmarking CUDA Gradient Synchronization")
    print("----------------------------------------")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Gradient size: {args.gradient_size}")
    print(f"Iterations: {args.iterations}")
    print("----------------------------------------")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
    else:
        print("CUDA is not available. Running on CPU.")
    
    # Run benchmarks
    custom_time = benchmark_custom_cuda_sync(
        num_gpus=args.num_gpus,
        gradient_size=args.gradient_size,
        iterations=args.iterations
    )
    
    print(f"\nResults:")
    print(f"Custom CUDA Sync time: {custom_time:.4f} seconds")
    if custom_time > 0:
        print(f"Average time per sync: {custom_time/args.iterations:.6f} seconds")
    print(f"Note: PyTorch DDP benchmark requires proper DDP setup")