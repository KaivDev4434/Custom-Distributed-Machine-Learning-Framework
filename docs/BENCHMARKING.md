# Benchmarking System

This document describes the benchmarking system for the parallel computing project.

## Directory Structure

```
benchmarks/
├── benchmark_stage1.py    # Data Loading Benchmark
├── benchmark_stage2.py    # CUDA Model Benchmark
├── benchmark_stage3.py    # MPI Gradient Sync Benchmark
└── benchmark_full.py      # Full Pipeline Benchmark

scripts/
└── launch_benchmarks.slurm  # Slurm script for HPC execution

results/
└── YYYYMMDD_HHMMSS/      # Results directory for each run
    ├── stage1_results.txt
    ├── stage2_results.txt
    ├── stage3_results.txt
    ├── full_pipeline_results.txt
    └── summary.txt
```

## Individual Stage Benchmarks

### Stage 1: Data Loading
- Compares custom C++/OpenMP data loader against PyTorch DataLoader
- Measures data loading and preprocessing time
- Command: `python benchmarks/benchmark_stage1.py`

### Stage 2: CUDA Model
- Benchmarks custom CUDA model against equivalent PyTorch model
- Measures forward pass execution time
- Command: `python benchmarks/benchmark_stage2.py`

### Stage 3: MPI Gradient Synchronization
- Tests custom MPI gradient synchronization
- Measures communication overhead
- Command: `srun --mpi=pmi2 python benchmarks/benchmark_stage3.py`

## Full Pipeline Benchmark

The full pipeline benchmark (`benchmark_full.py`) combines all stages and provides:
- End-to-end training performance
- Detailed profiling of each component
- Comparison with PyTorch DDP

Usage:
```bash
srun --mpi=pmi2 python benchmarks/benchmark_full.py [options]
```

Options:
- `--num-epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size for training (default: 64)
- `--profile`: Enable detailed profiling of each component

## Running on HPC

To run all benchmarks on an HPC system:

1. Submit the Slurm job:
```bash
sbatch scripts/launch_benchmarks.slurm
```

2. Monitor the job:
```bash
squeue -u $USER
```

3. View results:
```bash
cat results/YYYYMMDD_HHMMSS/*_results.txt
```

## Results Analysis

Each benchmark generates:
- Execution time for each component
- Speedup compared to PyTorch
- Detailed profiling data (when enabled)

Results are saved in timestamped directories under `results/` for easy comparison between runs.

## Requirements

### System Modules
- CUDA 12.4.0
- GCC 13.2.0
- foss/2023b (which includes MPI implementation)
- CMake 3.29.3

### Conda Environment
This project uses a conda environment named `apc_proj` with:
- PyTorch with CUDA support
- NumPy
- mpi4py
- Other Python dependencies

## Environment Setup

The benchmarking system uses both system modules and a conda environment:

### Module Loading
```bash
module load foss/2023b
module load CUDA/12.4.0
module load GCC/13.2.0
module load CMake/3.29.3
```

### Conda Environment 
```bash
# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate apc_proj
```

For custom configurations, edit the modules and environment in `scripts/launch_benchmarks.slurm`. 