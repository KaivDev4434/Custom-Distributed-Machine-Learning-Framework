#!/bin/bash
#SBATCH --job-name=mpi_gradient_sync
#SBATCH --output=logs/mpi_gradient_sync_%j.out
#SBATCH --error=logs/mpi_gradient_sync_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --qos=standard
#SBATCH --account=2025-spring-ds-642-bader-kd454
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load required modules
module purge
module load foss/2022b
module load GCC/12.2.0
module load CUDA/12.0.0
module load OpenMPI/4.1.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Add build directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# Print information about the job
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Starting at: $(date)"

# Run the MPI test
srun -p gpu \
     -n 1 \
     --ntasks-per-node=4 \
     --qos=standard \
     --account=2025-spring-ds-642-bader-kd454 \
     --gres=gpu:1 \
     python python/test_mpi.py

echo "Job completed at: $(date)" 