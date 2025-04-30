#!/bin/bash

# Usage instructions
if [ $# -lt 1 ]; then
    echo "Usage: $0 <num_processes>"
    echo "Example: $0 4"
    exit 1
fi

NUM_PROCESSES=$1

# Add build directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# Run the MPI test with srun
echo "Running MPI gradient sync test with $NUM_PROCESSES processes..."
srun -p gpu \
     -n 1 \
     --ntasks-per-node=$NUM_PROCESSES \
     --qos=standard \
     --account=2025-spring-ds-642-bader-kd454 \
     --gres=gpu:1 \
     --time=1:00:00 \
     python python/test_mpi.py

echo "Test completed!" 