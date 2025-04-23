import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def benchmark_pytorch(batch_size=64, hidden_size=128, epochs=5, num_workers=4, use_cuda=True):
    """Benchmark PyTorch model"""
    print(f"Benchmarking PyTorch model (batch_size={batch_size}, hidden_size={hidden_size})")
    
    # Set device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='../data/mnist',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Initialize model
    model = SimpleNN(
        input_size=28*28,
        hidden_size=hidden_size,
        num_classes=10
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    total_time = 0.0
    num_batches = min(100, len(train_loader))  # Limit to 100 batches for benchmarking
    
    # Warm-up
    print("Warming up...")
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 5:
            break
        
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start
        total_time += batch_time
        
        if (i + 1) % 10 == 0:
            print(f"Batch [{i+1}/{num_batches}], Time: {batch_time:.4f}s, Loss: {loss.item():.4f}")
    
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
    parser = argparse.ArgumentParser(description="Benchmark PyTorch neural network on MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    args = parser.parse_args()
    
    benchmark_pytorch(
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        use_cuda=not args.no_cuda
    )

if __name__ == "__main__":
    main() 