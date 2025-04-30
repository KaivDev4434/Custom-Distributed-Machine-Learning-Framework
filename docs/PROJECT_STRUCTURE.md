# Project Structure

This document outlines the organization of the parallel computing benchmarking project.

## Top-Level Directories

```
project/
├── benchmarks/      # Benchmarking scripts
├── build/           # Build artifacts (created by build script)
├── data/            # Data storage
├── docs/            # Documentation
├── logs/            # Log files from benchmark runs
├── results/         # Benchmark results
├── scripts/         # Build and run scripts
└── src/             # Source code
```

## Source Code Organization

The `src/` directory contains the implementation of our custom framework:

```
src/
├── data_loader/     # C++/OpenMP data loading implementation
│   ├── DataLoader.cpp
│   ├── DataLoader.h
│   └── bindings.cpp    # Python bindings
│
├── model/           # CUDA model implementation
│   ├── cuda_kernels.cu # CUDA kernel implementations
│   ├── Model.cpp       # C++ model wrapper
│   └── Model.h
│
├── mpi/             # MPI gradient synchronization
│   ├── gradient_sync.cu # CUDA-aware MPI implementation
│   └── bindings.cpp    # Python bindings
│
└── utils/           # Utility functions
    ├── timer.hpp       # Performance timing utilities
    └── checkpoint.cpp  # Model checkpoint utilities
```

## Benchmarking Scripts

The `benchmarks/` directory contains scripts for evaluating each stage of our implementation:

```
benchmarks/
├── benchmark_stage1.py  # Data loading benchmarks
├── benchmark_stage2.py  # CUDA model benchmarks
├── benchmark_stage3.py  # MPI gradient sync benchmarks
└── benchmark_full.py    # Full pipeline benchmarks
```

## Build and Run Scripts

The `scripts/` directory contains scripts for building and running the project:

```
scripts/
├── build.sh             # Build the C++/CUDA/MPI components
├── launch_benchmarks.slurm  # Slurm script for running benchmarks
├── run_all.sh           # Master script to build and launch benchmarks
└── collect_results_*.sh # Automatically generated result collection scripts
```

## Documentation

The `docs/` directory contains project documentation:

```
docs/
├── BENCHMARKING.md          # Documentation for the benchmarking system
├── main_implementation_plan.md  # Original project plan
└── PROJECT_STRUCTURE.md     # This file
```

## Results Organization

The `results/` directory contains timestamped directories for each benchmark run:

```
results/
└── YYYYMMDD_HHMMSS/      # Results from a specific run
    ├── stage1_results.txt
    ├── stage2_results.txt
    ├── stage3_results.txt
    ├── full_pipeline_results.txt
    └── summary.txt
``` 