#!/bin/bash

# Evaluation script - Compare results

set -e

echo "Comparing baseline and diffusion results..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(cd "$SCRIPT_DIR"/.. && pwd)"

cd "$PROJECT_DIR"

python -c "
import json
from pathlib import Path

def load_results(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

# Print results summary
print('\n' + '='*60)
print('EVALUATION RESULTS COMPARISON')
print('='*60)

baseline_log = Path('outputs/logs/baseline/baseline.log')
diffusion_log = Path('outputs/logs/diffusion/diffusion_pipeline.log')

if baseline_log.exists():
    print('\nBaseline results saved to: outputs/logs/baseline/')
    
if diffusion_log.exists():
    print('Diffusion results saved to: outputs/logs/diffusion/')

print('\n' + '='*60)
"

echo "Evaluation completed!"
