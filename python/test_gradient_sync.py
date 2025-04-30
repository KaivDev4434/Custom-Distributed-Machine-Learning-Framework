import numpy as np
import sys
sys.path.append('build/python')  # Add build directory to Python path
from mpi_wrapper import init_mpi, sync_gradients, finalize_mpi

def test_gradient_sync():
    # Initialize MPI
    rank = np.zeros(1, dtype=np.int32)
    size = np.zeros(1, dtype=np.int32)
    init_mpi(rank, size)
    
    # Create test gradients (different on each process)
    gradients = np.ones(10, dtype=np.float32) * (rank[0] + 1)
    print(f"Process {rank[0]}/{size[0]}, initial gradients: {gradients}")
    
    # Synchronize gradients
    sync_gradients(gradients)
    
    # Verify results
    expected = np.mean(np.arange(1, size[0] + 1)) * np.ones(10)
    print(f"Process {rank[0]}/{size[0]}, synchronized gradients: {gradients}")
    print(f"Process {rank[0]}/{size[0]}, expected gradients: {expected}")
    
    # Finalize MPI
    finalize_mpi()

if __name__ == "__main__":
    test_gradient_sync() 