#!/bin/bash
#SBATCH --job-name=test_gradient_sync
#SBATCH --output=logs/test_gradient_sync_%j.out
#SBATCH --error=logs/test_gradient_sync_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=your_partition_name  # Replace with your actual partition name

# Load required modules
module load python/3.x  # Replace with your Python version
module load mpi/openmpi-x.x.x  # Replace with your MPI version

# Run the test
srun python python/test_gradient_sync.py 