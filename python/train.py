import sys
import os
import time
import numpy as np
import argparse
from datetime import datetime

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

# Import our custom modules
import data_loader
import nn_model
from mpi_wrapper import init_mpi, finalize_mpi, get_world_size, get_world_rank, sync_gradients

def reshape_to_2d(images, batch_size):
    """Reshape images from [batch_size, 28, 28] to [batch_size, 784]"""
    return images.reshape(batch_size, -1)

def train_model(batch_size=64, hidden_size=128, num_epochs=10, learning_rate=0.01):
    # Initialize MPI
    init_mpi()
    world_size = get_world_size()
    rank = get_world_rank()
    
    # Initialize data loader
    print(f"Process {rank}: Initializing data loader...")
    custom_loader = data_loader.DataLoader(
        data_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "mnist")),
        batch_size=batch_size,
        num_workers=4
    )
    
    # Load MNIST data
    print(f"Process {rank}: Loading MNIST data...")
    custom_loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    num_samples = custom_loader.get_dataset_size()
    print(f"Process {rank}: Loaded {num_samples} training samples")
    
    # Initialize model
    print(f"Process {rank}: Initializing model...")
    model = nn_model.Model(
        input_size=28*28,  # MNIST images are 28x28
        hidden_size=hidden_size,
        num_classes=10     # MNIST has 10 classes (0-9)
    )
    
    # Training loop
    print(f"Process {rank}: Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        custom_loader.data_loader.dataset.indices = indices
        custom_loader.data_loader.dataset.indices_map = {i: indices[i] for i in range(num_samples)}
        
        # Process each batch
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_indices = custom_loader.data_loader.dataset.indices[i:batch_end]
            
            # Get next batch
            images, labels = custom_loader.get_next_batch()
            
            # Reshape images to [batch_size, 784]
            images = reshape_to_2d(images, batch_size)
            
            # Forward pass
            model.zero_grad()
            outputs = model.forward(images)
            
            # Compute loss and accuracy
            loss = model.compute_loss(outputs, labels)
            total_loss += loss
            
            # Backward pass
            gradients = model.backward(outputs, labels)
            
            # Synchronize gradients across processes
            sync_gradients(gradients)
            
            # Update weights
            model.update_weights(gradients, learning_rate)
            
            # Calculate accuracy
            predictions = np.argmax(outputs, axis=1)
            correct += np.sum(predictions == labels)
            total += len(labels)
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / (num_samples / batch_size)
        accuracy = correct / total
        
        if rank == 0:  # Only print from rank 0
            print(f"Process {rank}: Epoch {epoch + 1}/{num_epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
    
    # Finalize MPI
    finalize_mpi()
    
    # Save model
    if rank == 0:
        save_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.bin")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Process {rank}: Saving model to {save_path}")
        model.save(save_path)
        print(f"Process {rank}: Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Process {rank}: Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train neural network on MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--save-model", action="store_true", help="Save model after training")
    args = parser.parse_args()
    
    train_model(args.batch_size, args.hidden_size, args.epochs, args.learning_rate)

if __name__ == "__main__":
    main() 