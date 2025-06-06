#!/bin/bash

# Load modules and activate conda environment
module purge
module load wulver
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4
module load Miniforge3/24.1.2-0
module load CMake

# Use the correct path for conda.sh based on user's environment
CONDA_DIR="/mmfs1/apps/easybuild/software/Miniforge3/24.1.2-0"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
source "$CONDA_SH"

# Activate the Python 3.10 environment
conda activate py310

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# Check if number of processes was provided
NUM_PROCESSES=${1:-4}  # Default to 4 if not provided

# Create a wrapper script for srun
cat > scripts/srun_wrapper.sh << EOF
#!/bin/bash
source "$CONDA_SH"
conda activate py310
export PYTHONPATH=$PYTHONPATH
python "\$@"
EOF

chmod +x scripts/srun_wrapper.sh

# Run the test with srun
echo "Running MPI test with $NUM_PROCESSES processes..."
srun -p gpu \
    -n 1 \
    --ntasks-per-node=$NUM_PROCESSES \
    --qos=standard \
    --account=2025-spring-ds-642-bader-kd454 \
    --gres=gpu:1 \
    --time=1:00:00 \
    scripts/srun_wrapper.sh python/test_mpi.py

# Clean up
rm -f scripts/srun_wrapper.sh 