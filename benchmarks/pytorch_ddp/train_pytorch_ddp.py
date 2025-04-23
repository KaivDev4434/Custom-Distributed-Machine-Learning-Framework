import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

def train_ddp(rank, world_size, args):
    """Training function for DDP."""
    # Setup process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank}: Using device {device}")
    
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
    
    # Create distributed sampler
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = SimpleNN(
        input_size=28*28,
        hidden_size=args.hidden_size,
        num_classes=10
    ).to(device)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Training loop
    total_time = 0.0
    num_batches = min(100, len(train_loader))  # Limit to 100 batches for benchmarking
    
    # Warm-up
    if rank == 0:
        print("Warming up...")
    ddp_model.train()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 5:
            break
        
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        outputs = ddp_model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Benchmark
    if rank == 0:
        print(f"Benchmarking {num_batches} batches...")
    ddp_model.train()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    dist.barrier()
    
    start_time = time.time()
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = ddp_model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Synchronize before timing
        torch.cuda.synchronize()
        
        batch_time = time.time() - batch_start
        total_time += batch_time
        
        if (i + 1) % 10 == 0 and rank == 0:
            print(f"Batch [{i+1}/{num_batches}], Time: {batch_time:.4f}s, Loss: {loss.item():.4f}")
    
    # Final synchronization
    dist.barrier()
    
    avg_time = total_time / num_batches
    samples_per_sec = args.batch_size * world_size / avg_time
    
    if rank == 0:
        print(f"DDP Benchmark Results ({world_size} GPUs):")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average batch time: {avg_time:.4f}s")
        print(f"  Samples per second: {samples_per_sec:.2f}")
    
    # Cleanup
    dist.destroy_process_group()

def benchmark_pytorch_ddp(args):
    """Launch distributed benchmark."""
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No CUDA devices available. Running on CPU only.")
        world_size = 1
    
    print(f"Using {world_size} GPUs for distributed training")
    
    # Launch processes
    mp.spawn(
        train_ddp,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch DDP on MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads per GPU")
    args = parser.parse_args()
    
    benchmark_pytorch_ddp(args)

if __name__ == "__main__":
    main() 