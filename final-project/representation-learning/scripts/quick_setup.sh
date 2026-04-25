#!/bin/bash

#################################################################################
# Quick Setup Script
#
# This script runs the complete setup:
# 1. Installs dependencies
# 2. Downloads datasets
# 3. Verifies integrity
#
# Run this after cloning the repository
#################################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Representation Learning Project - Quick Setup             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Install dependencies
echo "Step 1/3: Installing dependencies..."
cd "$PROJECT_ROOT"
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 2: Setup datasets
echo "Step 2/3: Downloading and setting up datasets..."
bash "$PROJECT_ROOT/scripts/setup_datasets.sh"
echo ""

# Step 3: Create necessary directories
echo "Step 3/3: Creating working directories..."
mkdir -p "$PROJECT_ROOT"/{results/cifar10/{checkpoints,logs},results/credit_card_fraud/{checkpoints,logs}}
echo "✓ Directories created"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup Complete!                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Project structure:"
echo "  $(pwd)"
echo ""
echo "Ready to start development!"
