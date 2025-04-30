#!/bin/bash
# Master script for building and running benchmarks

# Set up error handling
set -e

# Load wulver first (required on this HPC system)
module purge
module load wulver

# Load Miniforge3 to access conda
module load Miniforge3

echo "=== Parallel Computing Benchmarking Project ==="
echo "Building and running all benchmarks"
echo "=============================================="

# Ensure conda environment is loaded
source $(conda info --base)/etc/profile.d/conda.sh
conda activate apc_proj
echo "Using conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Create necessary directories
mkdir -p logs
mkdir -p results

# Step 1: Build the project
echo "Step 1: Building the project..."
bash scripts/build.sh

# Step 2: Submit the benchmark job to Slurm
echo "Step 2: Submitting benchmark job to Slurm..."
JOB_ID=$(sbatch scripts/launch_benchmarks.slurm | awk '{print $4}')

echo "Job submitted with ID: $JOB_ID"
echo "Monitor job status with: squeue -j $JOB_ID"
echo "Results will be available in the results/ directory after completion"

# Step 3: Set up automatic result collection
echo "Step 3: Setting up result collection..."
cat > scripts/collect_results_${JOB_ID}.sh << EOF
#!/bin/bash
# Wait for job completion
while squeue -j ${JOB_ID} &>/dev/null; do
    echo "Job ${JOB_ID} is still running. Waiting 1 minute..."
    sleep 60
done

# Load wulver and other modules for the HPC environment
module purge
module load wulver

# Load Miniforge3 to access conda
module load Miniforge3

# Load conda environment for Python tools
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate apc_proj

# Create summary file
echo "Creating results summary..."
RESULT_DIR=\$(ls -td results/*/ | head -1)
echo "Results from job ${JOB_ID}" > \${RESULT_DIR}/summary.txt
echo "Run date: \$(date)" >> \${RESULT_DIR}/summary.txt
echo "" >> \${RESULT_DIR}/summary.txt

# Compile results
echo "== Data Loading Results ==" >> \${RESULT_DIR}/summary.txt
grep "Speedup:" \${RESULT_DIR}/stage1_results.txt >> \${RESULT_DIR}/summary.txt
echo "" >> \${RESULT_DIR}/summary.txt

echo "== CUDA Model Results ==" >> \${RESULT_DIR}/summary.txt
grep "Speedup:" \${RESULT_DIR}/stage2_results.txt >> \${RESULT_DIR}/summary.txt
echo "" >> \${RESULT_DIR}/summary.txt

echo "== CUDA Gradient Sync Results ==" >> \${RESULT_DIR}/summary.txt
grep "Custom MPI Sync time:" \${RESULT_DIR}/stage3_results.txt >> \${RESULT_DIR}/summary.txt
echo "" >> \${RESULT_DIR}/summary.txt

echo "== Full Pipeline Results ==" >> \${RESULT_DIR}/summary.txt
grep "Average time per epoch:" \${RESULT_DIR}/full_pipeline_results.txt >> \${RESULT_DIR}/summary.txt
echo "" >> \${RESULT_DIR}/summary.txt

# Generate performance visualization if matplotlib is available
python -c "
import matplotlib.pyplot as plt
import numpy as np
import re
import os

try:
    # Extract data
    with open('\${RESULT_DIR}/summary.txt', 'r') as f:
        content = f.read()
    
    # Extract speedups
    data_speedup = re.search(r'Speedup: (\\d+\\.\\d+)', content)
    model_speedup = re.search(r'Speedup: (\\d+\\.\\d+)', content[content.find('CUDA Model'):])
    
    if data_speedup and model_speedup:
        # Create bar chart
        labels = ['Data Loading', 'CUDA Model']
        values = [float(data_speedup.group(1)), float(model_speedup.group(1))]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['#3498db', '#2ecc71'])
        plt.axhline(y=1.0, color='r', linestyle='--', label='PyTorch baseline')
        
        # Add labels
        plt.title('Performance Comparison vs PyTorch', fontsize=16)
        plt.ylabel('Speedup Factor (higher is better)', fontsize=14)
        plt.ylim(bottom=0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}x', ha='center', fontsize=12)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('\${RESULT_DIR}/performance_comparison.png')
        print('Performance visualization saved to \${RESULT_DIR}/performance_comparison.png')
except Exception as e:
    print(f'Could not generate visualization: {e}')
"

conda deactivate
echo "For detailed results, see individual files in \${RESULT_DIR}"
echo "Results summary created: \${RESULT_DIR}/summary.txt"
EOF

chmod +x scripts/collect_results_${JOB_ID}.sh

# Deactivate conda environment
conda deactivate

echo "Result collection script created: scripts/collect_results_${JOB_ID}.sh"
echo "Run it after job completion to summarize results"
echo "=============================================="
echo "Setup complete! Check logs/benchmark_${JOB_ID}.out for progress" 