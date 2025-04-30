#!/bin/bash
#SBATCH --job-name=parallel_benchmarks
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=2025-spring-ds-642-bader-kd454

# Load wulver first (required on this HPC system)
module purge
module load wulver

# Load Miniforge3 to access conda
module load Miniforge3

# Load required modules - using compatible GCC version
module load foss/2022b
module load CUDA/12.0.0
module load GCC/12.2.0
module load CMake/3.24.3  # CMake version compatible with foss/2022b

# Activate conda environment for Python packages
source $(conda info --base)/etc/profile.d/conda.sh
conda activate apc_proj

echo "=== Environment Setup ==="
echo "Using modules:"
echo "  CUDA: $(nvcc --version | head -n 1)"
echo "  GCC: $(gcc --version | head -n 1)"
echo "  CMake: $(cmake --version | head -n 1)"
echo "Using conda environment: apc_proj"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")')"
echo "========================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Create results directory
RESULTS_DIR=results/$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

# Run individual stage benchmarks
echo "Running Stage 1 Benchmark (Data Loading)..."
python benchmarks/benchmark_stage1.py > $RESULTS_DIR/stage1_results.txt

echo "Running Stage 2 Benchmark (CUDA Model)..."
python benchmarks/benchmark_stage2.py > $RESULTS_DIR/stage2_results.txt

echo "Running Stage 3 Benchmark (CUDA Gradient Sync)..."
# Run directly since we're not using MPI
python benchmarks/benchmark_stage3.py > $RESULTS_DIR/stage3_results.txt

# Run full pipeline benchmark
echo "Running Full Pipeline Benchmark..."
python benchmarks/benchmark_full.py --num-epochs 5 --profile > $RESULTS_DIR/full_pipeline_results.txt

# Deactivate conda environment
conda deactivate 