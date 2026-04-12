#!/bin/bash
# Run all pipelines and compare results

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=================================="
echo "Running all pipelines..."
echo "=================================="

echo ""
echo "--- Baseline Pipeline ---"
python -m src.cli.run_baseline --config configs/default.yaml --output-dir outputs

echo ""
echo "--- GAN Pipeline ---"
python -m src.cli.run_gan --config configs/gan.yaml --output-dir outputs

echo ""
echo "--- VAE Pipeline ---"
python -m src.cli.run_vae --config configs/vae.yaml --output-dir outputs

echo ""
echo "=================================="
echo "All pipelines completed!"
echo "=================================="
echo "Results saved to outputs/"
