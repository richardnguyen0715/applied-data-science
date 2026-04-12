#!/bin/bash
# Run GAN-based oversampling pipeline

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Running GAN-based oversampling pipeline..."
python -m src.cli.run_gan --config configs/gan.yaml --output-dir outputs

echo "GAN pipeline completed!"
