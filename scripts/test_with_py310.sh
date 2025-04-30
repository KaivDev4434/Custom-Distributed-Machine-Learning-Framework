#!/bin/bash

# Source the Python 3.10 environment
source scripts/use_py310.sh

# Make sure the build directory is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# Run the simple test first
echo "Running simple MPI test..."
python python/simple_mpi_test.py

# If successful, run the full test with srun
if [ $? -eq 0 ]; then
    echo "Simple test successful, running with srun..."
    
    # Check if number of processes was provided
    NUM_PROCESSES=${1:-4}  # Default to 4 if not provided
    
    # Create a wrapper script for srun that activates the conda environment
    cat > scripts/py310_wrapper.sh << 'INNEREOF'
#!/bin/bash
source "$(dirname $(which conda))"/../etc/profile.d/conda.sh
conda activate py310
python "$@"
INNEREOF

    chmod +x scripts/py310_wrapper.sh
    
    echo "Running MPI test with $NUM_PROCESSES processes..."
    srun -p gpu \
        -n 1 \
        --ntasks-per-node=$NUM_PROCESSES \
        --qos=standard \
        --account=2025-spring-ds-642-bader-kd454 \
        --gres=gpu:1 \
        --time=1:00:00 \
        scripts/py310_wrapper.sh python/test_mpi.py
        
    # Clean up
    rm -f scripts/py310_wrapper.sh
else
    echo "Simple test failed, please fix the errors before running with srun."
fi
