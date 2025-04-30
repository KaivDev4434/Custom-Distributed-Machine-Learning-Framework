import numpy as np
import sys
import os

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

try:
    from mpi_wrapper_alt import init_mpi, finalize_mpi, sync_gradients, get_world_size, get_world_rank
    print("Successfully imported alternative MPI wrapper")
except ImportError as e:
    print(f"Error importing alternative MPI wrapper: {e}")
    print("Using regular MPI wrapper as fallback")
    from mpi_wrapper import init_mpi, finalize_mpi, sync_gradients, get_world_size, get_world_rank

def test_mpi_sync():
    # Initialize MPI
    rank = np.zeros(1, dtype=np.int32)
    size = np.zeros(1, dtype=np.int32)
    init_mpi(rank, size)
    
    # Get world size and rank from the arrays
    world_size = size[0]
    world_rank = rank[0]
    
    print(f"Process {world_rank}/{world_size} started")
    
    # Create test gradients
    gradients = np.ones(10, dtype=np.float32) * (world_rank + 1)  # Each process has different gradients
    
    print(f"Process {world_rank} initial gradients:", gradients)
    
    # Synchronize gradients (this returns a new array in the alt implementation)
    synced_gradients = sync_gradients(gradients)
    
    print(f"Process {world_rank} synchronized gradients:", synced_gradients)
    
    # Verify the result
    expected = np.ones(10, dtype=np.float32) * (world_size + 1) / 2  # Average of all processes
    if np.allclose(synced_gradients, expected):
        print(f"Process {world_rank}: Test passed!")
    else:
        print(f"Process {world_rank}: Test failed!")
        print(f"Expected: {expected}")
        print(f"Got: {synced_gradients}")
    
    # Finalize MPI
    finalize_mpi()

if __name__ == "__main__":
    test_mpi_sync() 