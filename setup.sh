#!/bin/bash
# Setup script for LiveVoice (inside docker: `docker attach yejin2 && conda activate sound`)

set -e

echo "Setting up LiveVoice environment..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing livevoice package..."
pip install -e .

echo "Creating output directories..."
mkdir -p output output/logs output/checkpoints

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
echo "✓ Setup complete. Example commands:"
echo "  python src/scripts/train_uncond.py --exp_name uncond_vctk --batch_size 16"
echo "  python src/scripts/train.py        --exp_name base_vctk   --batch_size 16"
