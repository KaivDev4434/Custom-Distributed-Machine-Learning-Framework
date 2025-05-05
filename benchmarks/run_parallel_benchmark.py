#!/usr/bin/env python3
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def run_benchmark(omp_threads, num_workers, batch_size=64):
    """Run benchmark_stage1.py with specified parameters and return results"""
    cmd = f"python3 benchmarks/benchmark_stage1.py --omp-threads {omp_threads} --num-workers {num_workers} --batch-size {batch_size}"
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    
    # Parse results
    custom_time = None
    torch_time = None
    speedup = None
    
    for line in output.split('\n'):
        if "Custom DataLoader time:" in line:
            custom_time = float(line.split(':')[1].strip().split()[0])
        elif "PyTorch DataLoader time:" in line:
            torch_time = float(line.split(':')[1].strip().split()[0])
        elif "Speedup:" in line and "N/A" not in line:
            speedup = float(line.split(':')[1].strip().split('x')[0])
    
    return {
        "omp_threads": omp_threads,
        "num_workers": num_workers,
        "custom_time": custom_time,
        "torch_time": torch_time,
        "speedup": speedup
    }

def main():
    # Thread configurations to test
    thread_configs = [
        {"omp_threads": 1, "num_workers": 1},
        {"omp_threads": 2, "num_workers": 2},
        {"omp_threads": 4, "num_workers": 4},
        {"omp_threads": 8, "num_workers": 8},
        {"omp_threads": 16, "num_workers": 16},
        {"omp_threads": 32, "num_workers": 32}
    ]
    
    results = []
    for config in thread_configs:
        result = run_benchmark(config["omp_threads"], config["num_workers"])
        results.append(result)
        print(f"OMP Threads: {result['omp_threads']}, Workers: {result['num_workers']}, "
              f"Custom Time: {result['custom_time']:.4f}s, Torch Time: {result['torch_time']:.4f}s, "
              f"Speedup: {result['speedup']:.2f}x\n")
    
    # Save results to CSV
    with open('benchmark_parallel_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['omp_threads', 'num_workers', 'custom_time', 'torch_time', 'speedup']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print("Results saved to benchmark_parallel_results.csv")
    
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
    plt.plot(thread_counts, custom_times, 'o-', label='Custom DataLoader')
    plt.plot(thread_counts, torch_times, 'o-', label='PyTorch DataLoader')
    plt.xlabel('Number of Threads')
    plt.ylabel('Time (seconds)')
    plt.title('DataLoader Execution Time vs Thread Count')
    plt.legend()
    plt.grid(True)
    
    # Plot speedup
    plt.subplot(2, 1, 2)
    plt.plot(thread_counts, speedups, 'o-', color='green')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (x)')
    plt.title('Speedup vs Thread Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_parallel_results.png')
    print("Plot saved as benchmark_parallel_results.png")

if __name__ == "__main__":
    main() 