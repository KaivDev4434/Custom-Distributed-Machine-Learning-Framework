#!/usr/bin/env python3
import time
import numpy as np
import sys
import os
import argparse

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "lib"))

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Stage 1: Data Loading')
    parser.add_argument('--omp-threads', type=int, default=1, 
                       help='Number of OpenMP threads to use')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of workers for DataLoader')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for DataLoader')
    return parser.parse_args()

def set_omp_threads(num_threads):
    # Set OpenMP threads
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    print(f"Set OMP_NUM_THREADS to {num_threads}")

# Parse args before importing modules that might use OpenMP
args = parse_args()
set_omp_threads(args.omp_threads)

import data_loader  # Import the compiled C++ module
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import datasets, transforms

def benchmark_custom_loader(batch_size=64, num_workers=1):
    # Provide required arguments to the DataLoader constructor
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    
    print(f"Data path: {data_path}")
    print(f"Using num_workers={num_workers}")
    
    try:
        print("Creating DataLoader instance...")
        loader = data_loader.DataLoader(data_path=data_path, batch_size=batch_size, num_workers=num_workers)
        
        print("Attempting to load MNIST files...")
        start_time = time.time()
        
        # First load MNIST files
        loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        print("MNIST files loaded successfully")
        
        print("Attempting to get batch...")
        batch = loader.get_next_batch()
        print(f"Batch retrieved: {len(batch[0])} images, {len(batch[1])} labels")
        
        end_time = time.time()
    except Exception as e:
        print(f"Error in custom loader: {e}")
        return 0.0
        
    return end_time - start_time

def benchmark_pytorch_loader(batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = TorchDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx == 100:  # Run for 100 batches
            break
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    print("Benchmarking Data Loading Stage")
    print("-------------------------------")
    print(f"OpenMP Threads: {args.omp_threads}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Workers: {args.num_workers}")
    print("-------------------------------")
    
    # Warm up
    print("Warming up...")
    benchmark_custom_loader(args.batch_size, args.num_workers)
    benchmark_pytorch_loader(args.batch_size, args.num_workers)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    custom_time = benchmark_custom_loader(args.batch_size, args.num_workers)
    torch_time = benchmark_pytorch_loader(args.batch_size, args.num_workers)
    
    print(f"\nResults:")
    print(f"Custom DataLoader time: {custom_time:.4f} seconds")
    print(f"PyTorch DataLoader time: {torch_time:.4f} seconds")
    
    if custom_time > 0:
        print(f"Speedup: {torch_time/custom_time:.2f}x")
    else:
        print("Speedup: N/A (custom loader failed)") 