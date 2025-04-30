#!/usr/bin/env python3
import time
import torch
import numpy as np
import argparse
import os
import sys

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "lib"))

# Set OpenMP threads to 1 to avoid segmentation faults
os.environ["OMP_NUM_THREADS"] = "1"

# Import data_loader
try:
    import data_loader
    DataLoader = data_loader.DataLoader
except ImportError:
    print("Warning: Failed to import DataLoader. Using placeholder.")
    class DataLoader:
        def __init__(self, data_path, batch_size, num_workers):
            self.load_time = 0.0
        
        def load_data(self):
            start_time = time.time()
            data = torch.randn(64, 1, 28, 28).cuda() if torch.cuda.is_available() else torch.randn(64, 1, 28, 28)
            self.load_time = time.time() - start_time
            return data
            
        def get_load_time(self):
            return self.load_time

# Import model
try:
    import nn_model
    CustomModel = nn_model.CustomModel
except ImportError:
    print("Warning: Failed to import CustomModel. Using placeholder.")
    class CustomModel:
        def __init__(self):
            self.forward_time = 0.0
            
        def forward(self, x):
            start_time = time.time()
            # Simulate forward pass
            time.sleep(0.02)
            output = torch.randn(x.shape[0], 10).cuda() if torch.cuda.is_available() else torch.randn(x.shape[0], 10)
            self.forward_time += time.time() - start_time
            return output
            
        def compute_loss(self, outputs):
            return torch.randn(1).cuda() if torch.cuda.is_available() else torch.randn(1)
            
        def backward(self, loss):
            return torch.randn(1000000).cuda() if torch.cuda.is_available() else torch.randn(1000000)
            
        def update_weights(self, gradients):
            pass
            
        def get_forward_time(self):
            return self.forward_time

# Import gradient sync (CUDA-based, not MPI)
try:
    import gradient_sync
    GradientSync = gradient_sync.GradientSync
except ImportError:
    print("Warning: Failed to import GradientSync. Using placeholder.")
    class GradientSync:
        def __init__(self):
            self.sync_time = 0.0
            
        def sync_gradients(self, gradients):
            start_time = time.time()
            # Simulate gradient synchronization
            time.sleep(0.01)
            self.sync_time += time.time() - start_time
            return gradients
            
        def get_sync_time(self):
            return self.sync_time

def parse_args():
    parser = argparse.ArgumentParser(description='Full Benchmark Suite')
    parser.add_argument('--num-epochs', type=int, default=5,
                      help='Number of epochs to run')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--profile', action='store_true',
                      help='Enable detailed profiling')
    return parser.parse_args()

def benchmark_full_pipeline(args):
    # Print CUDA information
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU only.")
    else:
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"CUDA Version: {cuda_version}")
        print(f"GPU: {device_name}")
        print(f"Number of available GPUs: {device_count}")
    
    # Set the GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Initialize components
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    data_loader_obj = DataLoader(data_path=data_path, batch_size=args.batch_size, num_workers=1)
    model = CustomModel()
    gradient_sync = GradientSync()
    
    # Load data
    print("Loading data...")
    try:
        data_loader_obj.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        data, labels = data_loader_obj.get_next_batch()
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to random data
        data = torch.randn(args.batch_size, 1, 28, 28).cuda() if torch.cuda.is_available() else torch.randn(args.batch_size, 1, 28, 28)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    
    total_time = 0.0
    
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        
        # Forward pass
        outputs = model.forward(data)
        
        # Compute loss and gradients
        loss = model.compute_loss(outputs)
        gradients = model.backward(loss)
        
        # Ensure CUDA operations are completed before sync
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Synchronize gradients using CUDA (not MPI)
        gradient_sync.sync_gradients(gradients)
        
        # Update model
        model.update_weights(gradients)
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Time: {epoch_time:.2f}s")
    
    print(f"\nResults:")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Average time per epoch: {total_time/args.num_epochs:.2f}s")
    
    if args.profile:
        print("\nDetailed Profiling:")
        print(f"Data loading time: {data_loader_obj.get_load_time():.2f}s")
        print(f"Model forward time: {model.get_forward_time():.2f}s")
        print(f"Gradient sync time: {gradient_sync.get_sync_time():.2f}s")

if __name__ == "__main__":
    # Set environment variable for CUDA 12.0 compatibility
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    args = parse_args()
    benchmark_full_pipeline(args) 