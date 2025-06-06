#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="parallel_computing",
    version="1.0.0",
    description="Custom parallel computing framework with C++/CUDA/MPI",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    package_dir={"": "src"},
    package_data={
        "": ["*.so", "*.dylib", "*.dll"],
    },
    install_requires=[
        "numpy",
        "torch",
        "mpi4py",
    ],
) 
