#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime

# Add the root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_command(cmd):
    """Run a command and return its output."""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error: {stderr.decode()}")
        return None
    return stdout.decode()

def run_benchmarks(args):
    """Run all benchmarks and collect results."""
    results = {}
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("Starting benchmarks")
    print("=" * 80)
    
    # 1. Custom CUDA model benchmark
    print("\n[1/3] Running Custom CUDA Model benchmark")
    custom_cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "train_custom.py"),
        f"--batch-size={args.batch_size}",
        f"--hidden-size={args.hidden_size}",
        f"--num-workers={args.num_workers}"
    ]
    custom_output = run_command(custom_cmd)
    if custom_output:
        results["custom"] = parse_benchmark_output(custom_output)
    
    # 2. PyTorch single-GPU benchmark
    print("\n[2/3] Running PyTorch single-GPU benchmark")
    pytorch_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "pytorch_ddp", "train_pytorch.py"),
        f"--batch-size={args.batch_size}",
        f"--hidden-size={args.hidden_size}",
        f"--num-workers={args.num_workers}"
    ]
    pytorch_output = run_command(pytorch_cmd)
    if pytorch_output:
        results["pytorch"] = parse_benchmark_output(pytorch_output)
    
    # 3. PyTorch DDP benchmark (if multiple GPUs available)
    if args.run_ddp:
        print("\n[3/3] Running PyTorch DDP benchmark")
        ddp_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "pytorch_ddp", "train_pytorch_ddp.py"),
            f"--batch-size={args.batch_size}",
            f"--hidden-size={args.hidden_size}",
            f"--num-workers={args.num_workers}"
        ]
        ddp_output = run_command(ddp_cmd)
        if ddp_output:
            results["pytorch_ddp"] = parse_benchmark_output(ddp_output)
    
    # Save results
    results_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compare results
    compare_results(results)
    
    print(f"\nResults saved to {results_file}")

def parse_benchmark_output(output):
    """Parse benchmark output to extract metrics."""
    lines = output.strip().split('\n')
    
    # Look for the results section
    result_lines = [line for line in lines if "Total time:" in line or "Average batch time:" in line or "Samples per second:" in line]
    
    result = {}
    for line in result_lines:
        if "Total time:" in line:
            result["total_time"] = float(line.split(":")[-1].strip().rstrip('s'))
        elif "Average batch time:" in line:
            result["avg_batch_time"] = float(line.split(":")[-1].strip().rstrip('s'))
        elif "Samples per second:" in line:
            result["samples_per_sec"] = float(line.split(":")[-1].strip())
    
    return result

def compare_results(results):
    """Compare benchmark results."""
    print("\n" + "=" * 80)
    print("Benchmark Results Comparison")
    print("=" * 80)
    
    # Check which benchmarks ran successfully
    available_benchmarks = list(results.keys())
    if not available_benchmarks:
        print("No benchmark results available for comparison.")
        return
    
    # Print header
    header = "Metric"
    for benchmark in available_benchmarks:
        header += f" | {benchmark.upper()}"
    print(header)
    print("-" * len(header))
    
    # Print metrics
    metrics = ["total_time", "avg_batch_time", "samples_per_sec"]
    for metric in metrics:
        row = metric
        for benchmark in available_benchmarks:
            if metric in results[benchmark]:
                value = results[benchmark][metric]
                row += f" | {value:.4f}"
            else:
                row += " | N/A"
        print(row)
    
    # Compare samples per second (higher is better)
    if "samples_per_sec" in results.get("custom", {}) and "samples_per_sec" in results.get("pytorch", {}):
        speedup = results["custom"]["samples_per_sec"] / results["pytorch"]["samples_per_sec"]
        print(f"\nCustom vs PyTorch speedup: {speedup:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Run and compare benchmarks")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--run-ddp", action="store_true", help="Run PyTorch DDP benchmark")
    parser.add_argument("--output-dir", type=str, default="../results", help="Output directory for results")
    args = parser.parse_args()
    
    run_benchmarks(args)

if __name__ == "__main__":
    main() 