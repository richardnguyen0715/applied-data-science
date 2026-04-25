#!/bin/bash

################################################################################
# VISUALIZATION PIPELINE TEST SCRIPT
# Validates the visualization pipeline setup
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
test_start() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}🧪 $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

test_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

test_info() {
    echo -e "${YELLOW}ℹ️  INFO${NC}: $1"
}

# Tests
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  VISUALIZATION PIPELINE TEST SUITE                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test 1: Python Installation
test_start "Python Installation"
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    test_pass "$python_version"
else
    test_fail "Python 3 not found"
    exit 1
fi

# Test 2: Required Packages
test_start "Required Packages"
required_packages=("numpy" "pandas" "matplotlib" "seaborn" "torch" "torchvision")
for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null)
        test_pass "$package ($version)"
    else
        test_fail "$package not installed"
    fi
done

# Test 3: Module Structure
test_start "Module Structure"
if [ -f "$PROJECT_ROOT/src/visualization/__init__.py" ]; then
    test_pass "src/visualization/__init__.py exists"
else
    test_fail "src/visualization/__init__.py missing"
fi

if [ -f "$PROJECT_ROOT/src/visualization/dataset_analyzer.py" ]; then
    test_pass "src/visualization/dataset_analyzer.py exists"
else
    test_fail "src/visualization/dataset_analyzer.py missing"
fi

if [ -f "$PROJECT_ROOT/src/visualization/config.py" ]; then
    test_pass "src/visualization/config.py exists"
else
    test_fail "src/visualization/config.py missing"
fi

# Test 4: Scripts
test_start "Bash Scripts"
if [ -f "$PROJECT_ROOT/scripts/visualize_datasets.sh" ]; then
    test_pass "scripts/visualize_datasets.sh exists"
else
    test_fail "scripts/visualize_datasets.sh missing"
fi

if [ -f "$PROJECT_ROOT/scripts/visualize_quick.sh" ]; then
    test_pass "scripts/visualize_quick.sh exists"
else
    test_fail "scripts/visualize_quick.sh missing"
fi

# Test 5: Documentation
test_start "Documentation"
if [ -f "$PROJECT_ROOT/VISUALIZATION_GUIDE.md" ]; then
    test_pass "VISUALIZATION_GUIDE.md exists"
else
    test_fail "VISUALIZATION_GUIDE.md missing"
fi

# Test 6: Directory Structure
test_start "Directory Structure"
dirs=(
    "src/visualization"
    "scripts"
    "results"
    "data"
    "logs"
)

for dir in "${dirs[@]}"; do
    if [ -d "$PROJECT_ROOT/$dir" ]; then
        test_pass "$dir/ exists"
    else
        test_fail "$dir/ missing"
    fi
done

# Test 7: Import Test
test_start "Module Import"
cd "$PROJECT_ROOT"
if python3 -c "from src.visualization.dataset_analyzer import DatasetAnalyzer; from src.visualization.config import VisualizationConfig" 2>/dev/null; then
    test_pass "All modules import successfully"
else
    test_fail "Module import failed"
fi

# Test 8: Dataset Availability
test_start "Dataset Availability"
if [ -d "$PROJECT_ROOT/data/cifar-10-batches-py" ]; then
    file_count=$(find "$PROJECT_ROOT/data/cifar-10-batches-py" -type f | wc -l)
    test_pass "CIFAR10 dataset found ($file_count files)"
else
    test_info "CIFAR10 dataset not found (will auto-download on first run)"
fi

if [ -f "$PROJECT_ROOT/data/creditcardfraud/creditcard.csv" ]; then
    csv_size=$(du -h "$PROJECT_ROOT/data/creditcardfraud/creditcard.csv" | cut -f1)
    test_pass "Fraud dataset found ($csv_size)"
else
    test_info "Fraud dataset not found (download with: bash scripts/download_fraud_dataset.sh)"
fi

# Test 9: Configuration Module
test_start "Configuration Module"
config_test=$(python3 << 'EOF'
from src.visualization.config import VisualizationConfig, get_default_config
config = VisualizationConfig()
print("OK" if config.get("cifar10", "num_sample_images") == 12 else "FAIL")
EOF
)
if [ "$config_test" = "OK" ]; then
    test_pass "Configuration module works"
else
    test_fail "Configuration module error"
fi

# Test 10: Permissions
test_start "Script Permissions"
if [ -x "$PROJECT_ROOT/scripts/visualize_datasets.sh" ]; then
    test_pass "visualize_datasets.sh is executable"
else
    test_fail "visualize_datasets.sh not executable"
    chmod +x "$PROJECT_ROOT/scripts/visualize_datasets.sh"
    test_info "Fixed: Made visualize_datasets.sh executable"
fi

if [ -x "$PROJECT_ROOT/scripts/visualize_quick.sh" ]; then
    test_pass "visualize_quick.sh is executable"
else
    test_fail "visualize_quick.sh not executable"
    chmod +x "$PROJECT_ROOT/scripts/visualize_quick.sh"
    test_info "Fixed: Made visualize_quick.sh executable"
fi

# Summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TEST SUMMARY                                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Ensure datasets are downloaded:"
    echo "     bash scripts/download_fraud_dataset.sh"
    echo ""
    echo "  2. Run visualization pipeline:"
    echo "     bash scripts/visualize_datasets.sh"
    echo ""
    echo "  3. Or quick analysis:"
    echo "     bash scripts/visualize_quick.sh"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please fix the issues above and run the tests again."
    echo ""
    exit 1
fi
