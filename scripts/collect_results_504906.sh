#!/bin/bash
# Wait for job completion
while squeue -j 504906 &>/dev/null; do
    echo "Job 504906 is still running. Waiting 1 minute..."
    sleep 60
done

# Load wulver and other modules for the HPC environment
module purge
module load wulver

# Load Miniforge3 to access conda
module load Miniforge3

# Load conda environment for Python tools
source $(conda info --base)/etc/profile.d/conda.sh
conda activate apc_proj

# Create summary file
echo "Creating results summary..."
RESULT_DIR=$(ls -td results/*/ | head -1)
echo "Results from job 504906" > ${RESULT_DIR}/summary.txt
echo "Run date: $(date)" >> ${RESULT_DIR}/summary.txt
echo "" >> ${RESULT_DIR}/summary.txt

# Compile results
echo "== Data Loading Results ==" >> ${RESULT_DIR}/summary.txt
grep "Speedup:" ${RESULT_DIR}/stage1_results.txt >> ${RESULT_DIR}/summary.txt
echo "" >> ${RESULT_DIR}/summary.txt

echo "== CUDA Model Results ==" >> ${RESULT_DIR}/summary.txt
grep "Speedup:" ${RESULT_DIR}/stage2_results.txt >> ${RESULT_DIR}/summary.txt
echo "" >> ${RESULT_DIR}/summary.txt

echo "== CUDA Gradient Sync Results ==" >> ${RESULT_DIR}/summary.txt
grep "Custom MPI Sync time:" ${RESULT_DIR}/stage3_results.txt >> ${RESULT_DIR}/summary.txt
echo "" >> ${RESULT_DIR}/summary.txt

echo "== Full Pipeline Results ==" >> ${RESULT_DIR}/summary.txt
grep "Average time per epoch:" ${RESULT_DIR}/full_pipeline_results.txt >> ${RESULT_DIR}/summary.txt
echo "" >> ${RESULT_DIR}/summary.txt

# Generate performance visualization if matplotlib is available
python -c "
import matplotlib.pyplot as plt
import numpy as np
import re
import os

try:
    # Extract data
    with open('${RESULT_DIR}/summary.txt', 'r') as f:
        content = f.read()
    
    # Extract speedups
    data_speedup = re.search(r'Speedup: (\d+\.\d+)', content)
    model_speedup = re.search(r'Speedup: (\d+\.\d+)', content[content.find('CUDA Model'):])
    
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
        plt.savefig('${RESULT_DIR}/performance_comparison.png')
        print('Performance visualization saved to ${RESULT_DIR}/performance_comparison.png')
except Exception as e:
    print(f'Could not generate visualization: {e}')
"

conda deactivate
echo "For detailed results, see individual files in ${RESULT_DIR}"
echo "Results summary created: ${RESULT_DIR}/summary.txt"
