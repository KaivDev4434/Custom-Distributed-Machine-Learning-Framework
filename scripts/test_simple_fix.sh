#!/bin/bash

# Load modules with explicit paths
module purge
module load wulver
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4
module load Miniforge3/24.1.2-0
module load CMake

# Use the correct path for conda.sh
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"

# Activate the py310 environment
echo "Activating Python 3.10 environment..."
conda activate py310

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# Number of processes to use
NUM_PROCESSES=${1:-4}  # Default to 4 if not provided

echo "Testing MPI gradient synchronization with $NUM_PROCESSES processes using Python wrapper..."

# Create a wrapper script for mpirun
cat > scripts/simple_wrapper.sh << 'EOF'
#!/bin/bash
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py310

# Add build directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $0))/build/python

exec python "$@"
EOF

chmod +x scripts/simple_wrapper.sh

# Run the test with mpirun
mpirun -np $NUM_PROCESSES scripts/simple_wrapper.sh python/simple_mpi_fix.py

# Clean up
rm -f scripts/simple_wrapper.sh 