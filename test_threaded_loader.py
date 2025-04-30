#!/usr/bin/env python3
import sys
import os
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Test DataLoader with different thread configurations')
    parser.add_argument('--omp-threads', type=int, default=4, 
                       help='Number of OpenMP threads to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for DataLoader')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for DataLoader')
    parser.add_argument('--debug', action='store_true',
                       help='Enable verbose debugging output')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set OpenMP threads
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    print(f"Set OMP_NUM_THREADS to {args.omp_threads}")
    
    # Add the build directory to Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib"))
    
    print(f"Attempting to import the data_loader module with {args.omp_threads} OpenMP threads and {args.num_workers} workers...")
    
    try:
        import data_loader
        print("Successfully imported data_loader module")
        
        if args.debug:
            print("Printing module info:", data_loader)
            print("DataLoader class:", data_loader.DataLoader)
        
        # Get MNIST data path
        data_path = os.path.join(os.path.dirname(__file__), "data", "mnist")
        print(f"Data path: {data_path}")
        
        if args.debug:
            print(f"Data exists: {os.path.exists(data_path)}")
            if os.path.exists(data_path):
                print(f"Contents: {os.listdir(data_path)}")
        
        # Create DataLoader with specified parameters
        print(f"Creating loader with batch_size={args.batch_size}, num_workers={args.num_workers}...")
        loader = data_loader.DataLoader(data_path=data_path, batch_size=args.batch_size, num_workers=args.num_workers)
        print("Successfully created DataLoader instance")
        
        # Try loading MNIST data
        print("\nLoading MNIST data...")
        start_time = time.time()
        loader.load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        load_time = time.time() - start_time
        print(f"Successfully loaded MNIST data in {load_time:.4f} seconds")
        
        # Try getting a batch
        print("\nGetting a batch...")
        batch_times = []
        for i in range(5):  # Get 5 batches
            start_time = time.time()
            images, labels = loader.get_next_batch()
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            print(f"Batch {i+1}: Retrieved {len(images)} images, {len(labels)} labels in {batch_time:.4f} seconds")
        
        # Print performance summary
        print(f"\nAverage batch retrieval time: {sum(batch_times)/len(batch_times):.4f} seconds")
        print(f"Dataset size: {loader.get_dataset_size()}")
        print(f"Batch size: {loader.get_batch_size()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 