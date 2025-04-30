#!/bin/bash
set -e

# Create a backup of environment.yml if it doesn't exist
if [ ! -f environment.yml.backup ]; then
    echo "Creating backup of environment.yml..."
    cp environment.yml environment.yml.backup
fi

# Create the conda environment
echo "Creating conda environment 'apc_proj'..."
conda env create -f environment.yml

# Activate the environment
echo "To activate the environment, run:"
echo "conda activate apc_proj"

echo "Setup complete!" 