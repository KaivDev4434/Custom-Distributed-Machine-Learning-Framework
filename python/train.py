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

def reshape_to_2d(images, batch_size):
    """Reshape images from [batch_size, 28, 28] to [batch_size, 784]"""
    return images.reshape(batch_size, -1)

def train(args):
    """Train the neural network"""
    print(f"Training model with hidden size {args.hidden_size}")
    
    # Initialize data loader
    print("Initializing data loader...")
    custom_loader = data_loader.DataLoader(
        data_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "mnist")),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load MNIST data
    print("Loading MNIST data...")
    custom_loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    num_samples = custom_loader.get_dataset_size()
    print(f"Loaded {num_samples} training samples")
    
    # Initialize model
    print("Initializing model...")
    model = nn_model.Model(
        input_size=28*28,  # MNIST images are 28x28
        hidden_size=args.hidden_size,
        num_classes=10     # MNIST has 10 classes (0-9)
    )
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    num_batches = num_samples // args.batch_size
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for i in range(num_batches):
            # Get next batch
            images, labels = custom_loader.get_next_batch()
            
            # Reshape images to [batch_size, 784]
            images = reshape_to_2d(images, args.batch_size)
            
            # Forward pass
            model.zero_grad()
            model.forward(images)
            
            # Backward pass
            model.backward(labels)
            
            # Update weights
            model.update(args.learning_rate)
            
            # Accumulate loss
            batch_loss = model.get_loss()
            epoch_loss += batch_loss
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{num_batches}], Loss: {batch_loss:.4f}")
        
        # End of epoch
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{args.epochs}] completed, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.bin")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving model to {save_path}")
        model.save(save_path)
    
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Train neural network on MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--save-model", action="store_true", help="Save model after training")
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main() 