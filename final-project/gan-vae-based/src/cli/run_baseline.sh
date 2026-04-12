#!/bin/bash
# Run baseline pipeline (no oversampling)

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Running baseline pipeline..."
python -m src.cli.run_baseline --config configs/default.yaml --output-dir outputs

echo "Baseline pipeline completed!"
