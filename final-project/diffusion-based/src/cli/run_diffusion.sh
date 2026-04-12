#!/bin/bash

# Diffusion-based oversampling pipeline script

set -e

echo "Running diffusion-based pipeline..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(cd "$SCRIPT_DIR"/.. && pwd)"

cd "$PROJECT_DIR"

python -m src.pipeline.run_diffusion_pipeline

echo "Diffusion pipeline completed!"
