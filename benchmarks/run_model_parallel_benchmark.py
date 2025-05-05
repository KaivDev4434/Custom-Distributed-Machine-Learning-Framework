#!/usr/bin/env python3
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import re

def run_benchmark(omp_threads):
    """Run benchmark_stage2.py with specified OMP_NUM_THREADS and return results"""
    # Set environment variable for OpenMP threads
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    
    cmd = f"python3 benchmarks/benchmark_stage2.py"
    print(f"Running benchmark with OMP_NUM_THREADS={omp_threads}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    output = result.stdout
    
    # Parse results
    custom_time = None
    torch_time = None
    speedup = None
    
    for line in output.split('\n'):
        if "Custom Model training time:" in line:
            match = re.search(r"([0-9]+\.[0-9]+)", line)
            if match:
                custom_time = float(match.group(1))
        elif "PyTorch Model training time:" in line:
            match = re.search(r"([0-9]+\.[0-9]+)", line)
            if match:
                torch_time = float(match.group(1))
        elif "Training speedup:" in line:
            match = re.search(r"([0-9]+\.[0-9]+)", line)
            if match:
                speedup = float(match.group(1))
    
    return {
        "omp_threads": omp_threads,
        "custom_time": custom_time,
        "torch_time": torch_time,
        "speedup": speedup
    }

def main():
    # Thread configurations to test
    thread_counts = [1, 2, 4, 8, 16, 32]
    
    results = []
    for threads in thread_counts:
        result = run_benchmark(threads)
        results.append(result)
        print(f"OMP Threads: {result['omp_threads']}, "
              f"Custom Time: {result['custom_time']:.4f}s, Torch Time: {result['torch_time']:.4f}s, "
              f"Speedup: {result['speedup']:.2f}x\n")
    
    # Save results to CSV
    with open('benchmark_model_parallel_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['omp_threads', 'custom_time', 'torch_time', 'speedup']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print("Results saved to benchmark_model_parallel_results.csv")
    
    # Create plots
    plot_results(results)

def plot_results(results):
    thread_counts = [r['omp_threads'] for r in results]
    custom_times = [r['custom_time'] for r in results]
    torch_times = [r['torch_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Plot execution times
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(thread_counts, custom_times, 'o-', label='Custom CUDA Model')
    plt.plot(thread_counts, torch_times, 'o-', label='PyTorch Model')
    plt.xlabel('Number of OpenMP Threads')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time vs OpenMP Thread Count')
    plt.legend()
    plt.grid(True)
    
    # Plot speedup
    plt.subplot(2, 1, 2)
    plt.plot(thread_counts, speedups, 'o-', color='green')
    plt.xlabel('Number of OpenMP Threads')
    plt.ylabel('Speedup (x)')
    plt.title('Speedup vs OpenMP Thread Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_model_parallel_results.png')
    print("Plot saved as benchmark_model_parallel_results.png")

if __name__ == "__main__":
    main() 