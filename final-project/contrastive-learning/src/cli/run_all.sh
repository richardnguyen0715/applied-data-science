#!/bin/bash

# Main script to run all experiments

echo "============================================"
echo "Contrastive Learning for Imbalanced Data"
echo "============================================"

# Option to run specific experiment
if [ -z "$1" ]; then
    echo "Usage: bash run_all.sh [cifar10lt|creditcard|all]"
    echo "  cifar10lt: Train on CIFAR-10-LT dataset"
    echo "  creditcard: Train on Credit Card Fraud Detection dataset"
    echo "  all: Run all experiments"
    exit 1
fi

EXPERIMENT=$1

if [ "$EXPERIMENT" = "cifar10lt" ] || [ "$EXPERIMENT" = "all" ]; then
    echo ""
    echo "Running CIFAR-10-LT experiment..."
    bash src/cli/train_cifar10lt.sh
fi

if [ "$EXPERIMENT" = "creditcard" ] || [ "$EXPERIMENT" = "all" ]; then
    echo ""
    echo "Running Credit Card Fraud Detection experiment..."
    bash src/cli/train_creditcard.sh
fi

echo ""
echo "============================================"
echo "All experiments completed!"
echo "============================================"
