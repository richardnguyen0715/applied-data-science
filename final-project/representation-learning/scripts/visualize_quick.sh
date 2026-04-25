#!/bin/bash

################################################################################
# QUICK VISUALIZATION SCRIPT
# Fast analysis of datasets (skip heavy computations)
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🚀 Quick Dataset Visualization${NC}"
echo ""

cd "$PROJECT_ROOT"

# Run with Python directly (faster than bash wrapper)
python3 << 'EOF'
from src.visualization.dataset_analyzer import DatasetAnalyzer
import json

analyzer = DatasetAnalyzer(data_dir="./data", results_dir="./results")

print("📊 Analyzing CIFAR10...")
cifar10 = analyzer.analyze_cifar10()

print("💳 Analyzing Fraud Detection...")
fraud = analyzer.analyze_fraud()

print("\n" + "="*70)
print("✅ VISUALIZATION COMPLETE")
print("="*70)

# Quick stats
if 'error' not in cifar10:
    print(f"\nCIFAR10:")
    print(f"  • Training samples: {cifar10.get('train_samples'):,}")
    print(f"  • Test samples: {cifar10.get('test_samples'):,}")
    print(f"  • Classes: {cifar10.get('num_classes')}")
else:
    print(f"⚠️  CIFAR10: {cifar10['error']}")

if 'error' not in fraud:
    print(f"\nCredit Card Fraud:")
    print(f"  • Total transactions: {fraud.get('total_transactions'):,}")
    fraud_pct = fraud.get('class_distribution', {}).get('fraud_percentage', 0)
    print(f"  • Fraud rate: {fraud_pct:.4f}%")
    print(f"  • Features: {fraud.get('total_features')}")
else:
    print(f"⚠️  Fraud: {fraud['error']}")

print("\n📁 Output files:")
print("  • results/visualizations/ (7 PNG files)")
print("  • results/statistics/ (2 JSON files)")
print("  • results/VISUALIZATION_REPORT.md")
print("")
EOF

echo -e "${GREEN}✅ Done!${NC}"
