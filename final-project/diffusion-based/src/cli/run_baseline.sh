#!/bin/bash

# Baseline evaluation script

set -e

echo "Running baseline evaluation..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(cd "$SCRIPT_DIR"/.. && pwd)"

cd "$PROJECT_DIR"

python -m src.pipeline.run_baseline

echo "Baseline evaluation completed!"
