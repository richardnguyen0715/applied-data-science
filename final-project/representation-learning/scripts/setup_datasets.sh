#!/bin/bash

#################################################################################
# Dataset Setup Pipeline
# 
# This script downloads and verifies both CIFAR10 and Credit Card Fraud
# Detection datasets for the Representation Learning project.
#
# Usage: ./scripts/setup_datasets.sh [OPTIONS]
#
# Options:
#   --skip-cifar10      Skip CIFAR10 download if already exists
#   --verify-only       Only verify existing datasets
#   --help              Show this help message
#################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Flags
SKIP_CIFAR10=false
VERIFY_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-cifar10)
            SKIP_CIFAR10=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --help)
            grep "^#" "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check Python
print_header "Checking Prerequisites"

if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed"
    exit 1
fi
print_success "Python3 found: $(python3 --version)"

# Check required packages
print_info "Checking required packages..."
python3 -c "import torch; import torchvision; import pandas" 2>/dev/null || {
    print_warning "Missing required packages"
    print_info "Installing dependencies..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
}
print_success "All required packages available"

# Setup datasets
print_header "Starting Dataset Setup"

cd "$PROJECT_ROOT"

# Build Python command
PYTHON_CMD="python3 src/preprocessing/download_datasets.py --project-root ."

if [ "$SKIP_CIFAR10" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip-cifar10"
fi

if [ "$VERIFY_ONLY" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --verify-only"
fi

# Run download script
if eval "$PYTHON_CMD"; then
    print_header "Setup Complete"
    print_success "Dataset pipeline completed successfully!"
    
    # Show next steps
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "  1. If using Credit Card Fraud dataset:"
    echo -e "     - Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo -e "     - Or use: kaggle datasets download -d mlg-ulb/creditcardfraud"
    echo -e "     - Extract to: data/credit_card_fraud/"
    echo -e "\n  2. Run verification: ./scripts/setup_datasets.sh --verify-only"
    echo -e "\n  3. Start training: python3 -m notebooks.train_cifar10"
    
    exit 0
else
    print_header "Setup Failed"
    print_error "Dataset pipeline encountered errors"
    exit 1
fi
