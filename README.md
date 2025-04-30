# Parallel Computing Benchmarking Project

A high-performance parallel computing implementation and benchmarking suite comparing custom C++/CUDA/MPI implementations against PyTorch DDP.

## Project Overview

This project implements a fully custom deep learning framework with:

1. **Custom Data Loading (C++/OpenMP)**: Parallelized data preprocessing
2. **CUDA Model**: Custom CUDA kernels for neural network operations
3. **Gradient Synchronization (CUDA/MPI)**: Multi-GPU communication using CUDA-aware MPI

The project benchmarks each component individually and as a combined pipeline against PyTorch's equivalent implementations.

## Directory Structure

```
├── benchmarks/                # Benchmarking scripts
│   ├── benchmark_stage1.py    # Data loading benchmarks
│   ├── benchmark_stage2.py    # CUDA model benchmarks
│   ├── benchmark_stage3.py    # MPI gradient sync benchmarks
│   └── benchmark_full.py      # Full pipeline benchmarks
│
├── data/                      # Data storage
│   └── mnist/                 # MNIST dataset
│
├── docs/                      # Documentation
│   ├── BENCHMARKING.md        # Benchmarking documentation
│   ├── PROJECT_STRUCTURE.md   # Project structure details
│   └── main_implementation_plan.md  # Project plan
│
├── logs/                      # Benchmark logs
│
├── results/                   # Benchmark results
│
├── scripts/                   # HPC scripts
│   ├── build.sh               # Build script
│   ├── run_all.sh             # All-in-one script
│   └── launch_benchmarks.slurm # Slurm job script
│
├── src/                       # Source code
│   ├── data_loader/           # C++/OpenMP data loader
│   ├── model/                 # CUDA model implementation
│   ├── mpi/                   # MPI gradient synchronization
│   └── utils/                 # Utility functions
│
└── README.md                  # This file
```

## Requirements

This project uses a combination of HPC modules and a conda environment:

### System Modules
- CUDA 12.4.0
- GCC 13.2.0
- foss/2023b (includes MPI)
- CMake 3.29.3

### Conda Environment
The project uses the `apc_proj` conda environment with:
- PyTorch with CUDA support
- NumPy
- mpi4py
- Other Python dependencies

## Building the Project

1. Make sure the conda environment is activated:
```bash
conda activate apc_proj
```

2. Build the C++/CUDA components:
```bash
./scripts/build.sh
```
This script automatically loads the required modules and uses the conda environment for Python bindings.

## Running Benchmarks

For the complete benchmark suite:
```bash
./scripts/run_all.sh
```

Or to submit just the benchmark job:
```bash
sbatch scripts/launch_benchmarks.slurm
```

To run individual benchmarks:
```bash
# Data loader benchmark
python benchmarks/benchmark_stage1.py

# Model benchmark
python benchmarks/benchmark_stage2.py

# MPI gradient sync benchmark (requires MPI)
srun --mpi=pmi2 python benchmarks/benchmark_stage3.py

# Full pipeline benchmark (requires MPI)
srun --mpi=pmi2 python benchmarks/benchmark_full.py --profile
```

## Results

Benchmark results are saved in the `results/` directory with timestamp-based directories for each run:
- Individual benchmark results
- Summary file with key metrics
- Performance visualization chart

## Documentation

- For detailed information on the benchmarking system, see [docs/BENCHMARKING.md](docs/BENCHMARKING.md)
- For project structure details, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) 