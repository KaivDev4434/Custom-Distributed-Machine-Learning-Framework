#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib"))

def verify_data_loader():
    print("Verifying data loader...")
    
    # Set OpenMP threads
    os.environ["OMP_NUM_THREADS"] = "4"
    
    try:
        import data_loader
        
        # Initialize the DataLoader
        data_path = os.path.join(os.path.dirname(__file__), "data", "mnist")
        loader = data_loader.DataLoader(data_path=data_path, batch_size=10, num_workers=4)
        
        # Load MNIST data
        print("Loading MNIST data...")
        loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        print(f"Dataset size: {loader.get_dataset_size()}")
        
        # Get a batch
        print("Getting a batch...")
        images, labels = loader.get_next_batch()
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        
        # Calculate statistics to verify normalization
        print(f"Image min value: {images.min()}")
        print(f"Image max value: {images.max()}")
        print(f"Image mean value: {images.mean()}")
        print(f"Image std value: {images.std()}")
        
        # Try saving one image from the batch for visual verification
        try:
            img = images[0].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(f"Label: {labels[0]}")
            plt.savefig("mnist_sample.png")
            print(f"Saved sample image to mnist_sample.png with label {labels[0]}")
        except Exception as e:
            print(f"Could not save image: {e}")
            
        return True
        
    except Exception as e:
        print(f"Error verifying data loader: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_data_loader() 