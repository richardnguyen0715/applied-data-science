#!/bin/bash

# Script to train contrastive learning model on Credit Card Fraud Detection

# Set parameters
DATASET="credit-card-fraud"
ENCODER_ARCHITECTURE="resnet18"
PROJECTION_DIM=128
HIDDEN_DIM=2048
CONTRASTIVE_EPOCHS=200
CLASSIFIER_EPOCHS=100
CONTRASTIVE_LR=0.5
CLASSIFIER_LR=0.001
BATCH_SIZE=512
TEMPERATURE=0.07
OUTPUT_DIR="outputs/credit-card-fraud"
SEED=42

echo "============================================"
echo "Training Contrastive Learning Model"
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "Note: Make sure to download creditcard.csv from Kaggle"
echo "and place it in data/creditcard.csv"
echo "============================================"

python main.py \
    --dataset "$DATASET" \
    --encoder-architecture "$ENCODER_ARCHITECTURE" \
    --projection-dim "$PROJECTION_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --contrastive-epochs "$CONTRASTIVE_EPOCHS" \
    --classifier-epochs "$CLASSIFIER_EPOCHS" \
    --contrastive-lr "$CONTRASTIVE_LR" \
    --classifier-lr "$CLASSIFIER_LR" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --device cuda

echo "Training completed!"
echo "Results saved to $OUTPUT_DIR"
