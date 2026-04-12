#!/bin/bash
# Run VAE-based oversampling pipeline

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Running VAE-based oversampling pipeline..."
python -m src.cli.run_vae --config configs/vae.yaml --output-dir outputs

echo "VAE pipeline completed!"
