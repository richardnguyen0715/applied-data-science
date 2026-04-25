#!/bin/bash
# Quick runner for imbalance handling comparison pipeline

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/src/configs/config.yaml"
RESULTS_DIR="${PROJECT_ROOT}/results/cifar10"
OUTPUT_DIR="${PROJECT_ROOT}/results/comparison"
NOTEBOOK_PATH="${PROJECT_ROOT}/notebooks/comparison_pipeline.ipynb"

# Activate virtual environment if available
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

# Parse arguments
MODE="${1:-notebook}"
case "$MODE" in
    notebook)
        echo -e "${BLUE}Opening comparison pipeline notebook...${NC}"
        if command -v jupyter &> /dev/null; then
            jupyter notebook "$NOTEBOOK_PATH"
        else
            echo -e "${YELLOW}⚠ Jupyter not found. Installing...${NC}"
            pip install jupyter
            jupyter notebook "$NOTEBOOK_PATH"
        fi
        ;;
    script)
        echo -e "${BLUE}Running comparison pipeline script...${NC}"
        python -m src.scripts.run_comparison \
            --config "$CONFIG_PATH" \
            --results-dir "$RESULTS_DIR" \
            --output-dir "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Comparison complete${NC}"
        echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
        ;;
    view-results)
        echo -e "${BLUE}Comparison Results${NC}"
        echo "=================================="
        if [[ -f "$OUTPUT_DIR/logs/comparison.log" ]]; then
            echo -e "${GREEN}Latest Report:${NC}"
            tail -50 "$OUTPUT_DIR/logs/comparison.log"
        else
            echo "No results found. Run comparison first."
        fi
        ;;
    help|*)
        cat << EOF
${BLUE}Imbalance Handling Comparison Pipeline${NC}

Usage: $0 [MODE]

Modes:
  notebook      Open interactive Jupyter notebook (default)
  script        Run comparison pipeline as script
  view-results  View latest comparison results
  help          Show this help message

Examples:
  # Run interactive notebook
  $0 notebook

  # Run as script (background friendly)
  $0 script

  # View latest results
  $0 view-results

Output files:
  - $OUTPUT_DIR/per_class_performance.csv
  - $OUTPUT_DIR/overall_metrics.csv
  - $OUTPUT_DIR/training_curves.png
  - $OUTPUT_DIR/per_class_accuracy.png
  - $OUTPUT_DIR/imbalance_vs_accuracy.png
  - $OUTPUT_DIR/logs/comparison.log

Documentation: docs/COMPARISON_PIPELINE.md
EOF
        ;;
esac
