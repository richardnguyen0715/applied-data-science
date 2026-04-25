#!/bin/bash

#################################################################################
# Kaggle Dataset Download Helper
#
# This script helps download the Credit Card Fraud dataset using Kaggle CLI
#
# Setup:
#   1. Install kaggle: pip install kaggle
#   2. Create Kaggle API token: https://www.kaggle.com/settings/account
#   3. Place kaggle.json in ~/.kaggle/
#   4. Run this script
#################################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="$PROJECT_ROOT/data/credit_card_fraud"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Kaggle Dataset Download Helper                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "✗ Kaggle CLI not found"
    echo "Install it with: pip install kaggle"
    exit 1
fi

echo "✓ Kaggle CLI found"
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
echo "Downloading Credit Card Fraud dataset to: $DATA_DIR"
echo ""

# Download dataset
cd "$DATA_DIR"
kaggle datasets download -d mlg-ulb/creditcardfraud

# Unzip
if [ -f creditcardfraud.zip ]; then
    echo "Extracting dataset..."
    unzip -q creditcardfraud.zip
    rm creditcardfraud.zip
    echo "✓ Dataset extracted successfully"
else
    echo "✗ Download failed"
    exit 1
fi

echo ""
echo "✓ Credit Card Fraud dataset ready!"
echo "Location: $DATA_DIR/creditcard.csv"
echo ""
echo "Run setup verification:"
echo "  ./scripts/setup_datasets.sh --verify-only"
