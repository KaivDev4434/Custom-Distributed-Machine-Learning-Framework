# Custom Deep Learning Framework

This project implements a custom deep learning framework with parallel data loading capabilities using C++/OpenMP, CUDA, and MPI.

## Stage 1: Custom Data Loader

The first stage implements a custom data loader for MNIST dataset using C++ and OpenMP for parallel data loading and preprocessing.

### Dependencies

- C++17 compatible compiler
- OpenMP
- Python 3.x
- pybind11
- NumPy
- PyTorch (for benchmarking)

### Building the Project

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build the project:
```bash
./scripts/build.sh
```

### Testing the Data Loader

To test the custom data loader and compare it with PyTorch's implementation:

```bash
python python/test_data_loader.py
```

This will:
1. Load MNIST data using both custom and PyTorch data loaders
2. Process 10 batches from each loader
3. Compare the performance

### Project Structure

```
your_project/
├── data/
│   └── mnist/               # MNIST dataset files
├── src/
│   └── data_loader/         # Custom data loading implementation
│       ├── DataLoader.cpp   # Main implementation
│       ├── DataLoader.h     # Header file
│       └── bindings.cpp     # Python bindings
├── python/
│   └── test_data_loader.py  # Test script
├── scripts/
│   └── build.sh            # Build script
└── docs/
    └── README.md           # Documentation
```

### Implementation Details

The custom data loader:
- Uses OpenMP for parallel data loading and preprocessing
- Implements efficient memory management
- Provides a Python interface through pybind11
- Supports batch processing and data shuffling
- Normalizes images during loading

### Next Steps

After completing Stage 1, the project will move on to:
1. Implementing CUDA kernels for model operations
2. Adding MPI support for distributed training
3. Benchmarking against PyTorch DDP 