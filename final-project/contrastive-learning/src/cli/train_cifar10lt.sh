#!/bin/bash

# Script to train contrastive learning model on CIFAR-10-LT

# Set parameters
DATASET="cifar10-lt"
CIFAR10_CONFIG="r-100"
ENCODER_ARCHITECTURE="resnet18"
PROJECTION_DIM=128
HIDDEN_DIM=2048
CONTRASTIVE_EPOCHS=200
CLASSIFIER_EPOCHS=100
CONTRASTIVE_LR=0.5
CLASSIFIER_LR=0.001
BATCH_SIZE=512
TEMPERATURE=0.07
OUTPUT_DIR="outputs/cifar10-lt"
SEED=42

echo "============================================"
echo "Training Contrastive Learning Model"
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "============================================"

python main.py \
    --dataset "$DATASET" \
    --cifar10-config "$CIFAR10_CONFIG" \
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
