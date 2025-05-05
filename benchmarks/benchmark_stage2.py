#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
import sys
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "build", "lib"))
import nn_model as model_module
import data_loader as data_loader_module

# Ensure CUDA compatibility
CUDA_VERSION = torch.version.cuda
print(f"Using CUDA version: {CUDA_VERSION}")

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        # Updated to match the custom model: simple MLP with 784->128->10
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 784)
        # FC1 with ReLU
        x = torch.relu(self.fc1(x))
        # FC2 (no activation - will be applied in loss function)
        x = self.fc2(x)
        return x

def train_pytorch_model(model, train_loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return accuracy, avg_loss

def train_custom_model(model, train_loader, learning_rate):
    correct = 0
    total = 0
    total_loss = 0
    
    # Calculate number of batches
    dataset_size = train_loader.get_dataset_size()
    batch_size = train_loader.get_batch_size()
    num_batches = dataset_size // batch_size
    
    for _ in range(num_batches):
        images, labels = train_loader.get_next_batch()
        images = images.reshape(-1, 784)  # Flatten images - this is now required
        
        # Forward pass
        model.forward(images)
        
        # Get predictions
        output = model.get_output()
        predictions = np.argmax(output, axis=1)
        
        # Calculate accuracy
        correct += np.sum(predictions == labels)
        total += len(labels)
        
        # Backward pass
        model.backward(images, labels)  # Now requires both images and labels
        model.update(learning_rate)
        model.zero_grad()
        
        # Get loss
        total_loss += model.get_loss()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / num_batches
    return accuracy, avg_loss

def benchmark_custom_model():
    # Create model with input channels=1 and batch_size=64
    model = model_module.Model(1, 64)
    
    # Print model architecture
    print("\nCustom Model Architecture:")
    print("Input: 784 → FC1(128) → ReLU → FC2(10) → Output")
    
    # Create custom data loader
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    data_loader = data_loader_module.DataLoader(data_path, batch_size=64)
    data_loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    
    # Training parameters
    learning_rate = 0.01
    num_epochs = 5
    
    # Record training time
    start_time = time.time()
    
    # Training loop
    print("\nTraining Custom Model:")
    for epoch in range(num_epochs):
        accuracy, _ = train_custom_model(model, data_loader, learning_rate)
        print(f"Epoch {epoch+1}/{num_epochs} - Accuracy: {accuracy:.2f}%")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Skip inference benchmarking, just return training time
    return training_time

def benchmark_pytorch_model():
    # Create model and move to GPU
    model = PyTorchModel().cuda()
    
    # Print model architecture
    print("\nPyTorch Model Architecture:")
    print(model)
    
    # Create PyTorch data loader using local files
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create custom dataset class to load from local files
    class LocalMNIST(torch.utils.data.Dataset):
        def __init__(self, data_path, transform=None):
            self.data_path = data_path
            self.transform = transform
            self.data_loader = data_loader_module.DataLoader(data_path, batch_size=1)
            self.data_loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
            self.length = self.data_loader.get_dataset_size()
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            images, labels = self.data_loader.get_next_batch()
            image = images[0].reshape(28, 28)  # Reshape to 28x28
            label = torch.tensor(labels[0], dtype=torch.long)  # Convert to long integer
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    train_dataset = LocalMNIST(data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Training parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Record training time
    start_time = time.time()
    
    # Training loop
    print("\nTraining PyTorch Model:")
    for epoch in range(5):
        accuracy, _ = train_pytorch_model(model, train_loader, optimizer, criterion, torch.device('cuda'))
        print(f"Epoch {epoch+1}/5 - Accuracy: {accuracy:.2f}%")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Skip inference benchmarking, just return training time
    return training_time

if __name__ == "__main__":
    print("Benchmarking Model Stage")
    print("-----------------------")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation.")
        exit(1)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    try:
        custom_time = benchmark_custom_model()
        print(f"Custom Model training time: {custom_time:.4f} seconds")
        
        torch_time = benchmark_pytorch_model()
        print(f"PyTorch Model training time: {torch_time:.4f} seconds")
        
        if custom_time > 0:
            print(f"Training speedup: {torch_time/custom_time:.2f}x")
        else:
            print("Could not calculate speedup due to timing error")
            
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()