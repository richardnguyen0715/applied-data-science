#!/bin/bash

#################################################################################
# Kagglehub Dataset Download Helper
#
# This script helps download the Credit Card Fraud dataset using kagglehub
#
# Setup:
#   1. Install kagglehub: pip install kagglehub
#   2. Create Kaggle API token: https://www.kaggle.com/settings/account
#   3. Place kaggle.json in ~/.kaggle/
#   4. Run this script
#################################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="$PROJECT_ROOT/data/credit_card_fraud"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Kagglehub Dataset Download Helper                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if kagglehub is installed
python3 -c "import kagglehub" 2>/dev/null || {
    echo "✗ kagglehub not found"
    echo "Install it with: pip install kagglehub"
    exit 1
}

echo "✓ kagglehub found"
echo ""

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "✗ Kaggle credentials not found"
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to: https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New API Token' (downloads kaggle.json)"
    echo "  3. Move kaggle.json to ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo "  5. Run this script again"
    exit 1
fi

echo "✓ Kaggle credentials found"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
echo "Downloading Credit Card Fraud dataset using kagglehub..."
echo "Target: $DATA_DIR"
echo ""

# Download using kagglehub
python3 << 'PYTHON'
import kagglehub
from pathlib import Path
import shutil

print("Downloading dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(f"✓ Dataset downloaded to: {path}")

# Copy CSV to our directory
source_csv = Path(path) / 'creditcard.csv'
target_csv = Path("'$DATA_DIR'") / 'creditcard.csv'

if source_csv.exists():
    print(f"Copying creditcard.csv...")
    shutil.copy(source_csv, target_csv)
    print(f"✓ File copied to: {target_csv}")
    print(f"✓ File size: {target_csv.stat().st_size / (1024*1024):.1f} MB")
else:
    print(f"✗ creditcard.csv not found in {path}")
    exit(1)
PYTHON

if [ $? -ne 0 ]; then
    echo "✗ Download failed"
    exit 1
fi

echo ""
echo "✓ Credit Card Fraud dataset ready!"
echo "Location: $DATA_DIR/creditcard.csv"
echo ""
echo "Run verification:"
echo "  ./scripts/setup_datasets.sh --verify-only"
