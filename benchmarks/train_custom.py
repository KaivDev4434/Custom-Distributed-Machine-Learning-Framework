import sys
import os
import time
import numpy as np
import argparse

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

# Import our custom modules
import data_loader
import nn_model

def benchmark_custom_model(batch_size=64, hidden_size=128, epochs=5, num_workers=4):
    """Benchmark our custom CUDA model"""
    print(f"Benchmarking custom model (batch_size={batch_size}, hidden_size={hidden_size})")
    
    # Initialize data loader
    custom_loader = data_loader.DataLoader(
        data_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "mnist")),
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Load MNIST data
    custom_loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    num_samples = custom_loader.get_dataset_size()
    
    # Initialize model
    model = nn_model.Model(
        input_size=28*28,  # MNIST images are 28x28
        hidden_size=hidden_size,
        num_classes=10     # MNIST has 10 classes (0-9)
    )
    
    # Training loop
    total_time = 0.0
    num_batches = min(100, num_samples // batch_size)  # Limit to 100 batches for benchmarking
    
    # Warm-up
    print("Warming up...")
    for i in range(5):
        images, labels = custom_loader.get_next_batch()
        images = images.reshape(batch_size, -1)
        model.forward(images)
        model.backward(labels)
        model.update(0.01)
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    
    for i in range(num_batches):
        batch_start = time.time()
        
        # Get next batch
        images, labels = custom_loader.get_next_batch()
        images = images.reshape(batch_size, -1)
        
        # Forward pass
        model.zero_grad()
        model.forward(images)
        
        # Backward pass
        model.backward(labels)
        
        # Update weights
        model.update(0.01)
        
        batch_time = time.time() - batch_start
        total_time += batch_time
        
        if (i + 1) % 10 == 0:
            print(f"Batch [{i+1}/{num_batches}], Time: {batch_time:.4f}s, Loss: {model.get_loss():.4f}")
    
    avg_time = total_time / num_batches
    samples_per_sec = batch_size / avg_time
    
    print(f"Benchmark Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average batch time: {avg_time:.4f}s")
    print(f"  Samples per second: {samples_per_sec:.2f}")
    
    return {
        "total_time": total_time,
        "avg_batch_time": avg_time,
        "samples_per_sec": samples_per_sec
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark custom neural network on MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    args = parser.parse_args()
    
    benchmark_custom_model(
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main() 