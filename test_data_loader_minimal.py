#!/usr/bin/env python3
import sys
import os

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib"))

try:
    print("Attempting to import the data_loader module...")
    import data_loader
    print("Successfully imported data_loader module")
    
    print("Printing module info:", data_loader)
    
    # Try to get the DataLoader class
    print("DataLoader class:", data_loader.DataLoader)
    
    # Get MNIST data path
    data_path = os.path.join(os.path.dirname(__file__), "data", "mnist")
    print(f"Data path: {data_path}")
    print(f"Data exists: {os.path.exists(data_path)}")
    
    if os.path.exists(data_path):
        print(f"Contents: {os.listdir(data_path)}")
    
    print("\nTrying to create loader with reduced OpenMP threads...")
    # Set OMP_NUM_THREADS to 1 to avoid threading issues
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Try creating the DataLoader with just one parameter at a time
    print("Creating loader with minimal args...")
    loader = data_loader.DataLoader(data_path=data_path, batch_size=64, num_workers=1)
    print("Successfully created DataLoader instance")
    
    # Try loading MNIST data
    print("\nLoading MNIST data...")
    loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    print("Successfully loaded MNIST data")
    
    # Try getting a batch
    print("\nGetting a batch...")
    images, labels = loader.get_next_batch()
    print(f"Successfully retrieved batch: {len(images)} images, {len(labels)} labels")
    
    # Print some stats
    print(f"\nDataset size: {loader.get_dataset_size()}")
    print(f"Batch size: {loader.get_batch_size()}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 