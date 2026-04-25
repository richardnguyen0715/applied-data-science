#!/bin/bash

################################################################################
# DATASET VISUALIZATION PIPELINE
# Automatically generates statistical visualizations for CIFAR10 and 
# Credit Card Fraud Detection datasets
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"
PYTHON_SCRIPT="$PROJECT_ROOT/src/visualization/dataset_analyzer.py"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}ℹ️  INFO${NC}: $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS${NC}: $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  WARNING${NC}: $1"
}

log_error() {
    echo -e "${RED}❌ ERROR${NC}: $1"
}

print_header() {
    echo ""
    echo "================================================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "================================================================================================"
    echo ""
}

print_footer() {
    echo ""
    echo "================================================================================================"
    echo -e "${GREEN}$1${NC}"
    echo "================================================================================================"
    echo ""
}

################################################################################
# Pre-flight Checks
################################################################################

print_header "🔍 DATASET VISUALIZATION PIPELINE"

log_info "Performing pre-flight checks..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    exit 1
fi
log_success "Python 3 found: $(python3 --version)"

# Check if required Python packages are installed
log_info "Checking required Python packages..."
python3 -c "import matplotlib" 2>/dev/null || {
    log_error "matplotlib not installed"
    exit 1
}
python3 -c "import seaborn" 2>/dev/null || {
    log_error "seaborn not installed"
    exit 1
}
python3 -c "import pandas" 2>/dev/null || {
    log_error "pandas not installed"
    exit 1
}
python3 -c "import numpy" 2>/dev/null || {
    log_error "numpy not installed"
    exit 1
}
log_success "All required packages are installed"

# Check if datasets exist
log_info "Checking dataset availability..."

if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    log_warning "CIFAR10 dataset not found at $DATA_DIR/cifar-10-batches-py"
    log_info "The pipeline will attempt to download CIFAR10 during analysis"
fi

if [ ! -f "$DATA_DIR/creditcardfraud/creditcard.csv" ]; then
    log_warning "Fraud dataset not found at $DATA_DIR/creditcardfraud/creditcard.csv"
    log_info "Please ensure the fraud dataset is downloaded using:"
    log_info "  bash scripts/download_fraud_dataset.sh"
fi

# Create output directories
log_info "Creating output directories..."
mkdir -p "$RESULTS_DIR/visualizations"
mkdir -p "$RESULTS_DIR/statistics"
mkdir -p "$PROJECT_ROOT/logs"
log_success "Output directories ready"

################################################################################
# Run Visualization Pipeline
################################################################################

print_header "🚀 RUNNING VISUALIZATION ANALYSIS"

cd "$PROJECT_ROOT" || exit 1

log_info "Executing dataset analyzer..."
log_info "This may take several minutes depending on dataset size..."
echo ""

if python3 "$PYTHON_SCRIPT"; then
    log_success "Visualization pipeline completed"
else
    log_error "Visualization pipeline failed"
    exit 1
fi

################################################################################
# Summary
################################################################################

print_footer "📊 VISUALIZATION COMPLETE"

echo -e "${BLUE}Output Summary:${NC}"
echo ""
echo "📁 Visualizations:"
echo "   - results/visualizations/cifar10_class_distribution.png"
echo "   - results/visualizations/cifar10_sample_images.png"
echo "   - results/visualizations/cifar10_channel_statistics.png"
echo "   - results/visualizations/fraud_class_distribution.png"
echo "   - results/visualizations/fraud_amount_distribution.png"
echo "   - results/visualizations/fraud_correlation_heatmap.png"
echo "   - results/visualizations/fraud_time_patterns.png"
echo ""
echo "📊 Statistics (JSON):"
echo "   - results/statistics/cifar10_statistics.json"
echo "   - results/statistics/fraud_statistics.json"
echo ""
echo "📄 Report:"
echo "   - results/VISUALIZATION_REPORT.md"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. View the report: cat results/VISUALIZATION_REPORT.md"
echo "2. View visualizations: open results/visualizations/"
echo "3. Check statistics: cat results/statistics/*.json"
echo ""
