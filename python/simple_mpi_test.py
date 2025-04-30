import numpy as np
import sys
import os

# Add the build directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

try:
    # Try to import just the basic MPI functions
    from mpi_wrapper import init_mpi, finalize_mpi
    print("Successfully imported basic MPI functions")
    
    # Initialize MPI
    rank = np.zeros(1, dtype=np.int32)
    size = np.zeros(1, dtype=np.int32)
    init_mpi(rank, size)
    
    print(f"Process rank: {rank[0]}, total processes: {size[0]}")
    
    # Finalize MPI
    finalize_mpi()
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}") 