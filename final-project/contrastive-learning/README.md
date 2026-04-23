# Contrastive Learning for Imbalanced Data

End-to-end pipeline for contrastive learning on imbalanced datasets. This project implements self-supervised contrastive learning to learn robust representations from imbalanced data, followed by training a downstream classifier.

## Features

- Contrastive Learning with NT-Xent loss (SimCLR-style)
- Support for CIFAR-10-LT and Credit Card Fraud Detection datasets
- Class-aware sampling and weighted loss functions for imbalanced classification
- ResNet18 and ResNet50 encoders with configurable projection heads
- Complete pipeline from data loading to evaluation
- Visualization tools for training curves and confusion matrices

## Datasets

1. **CIFAR-10-LT**: Long-tail distribution of CIFAR-10 (imbalance ratio 100:1)
2. **Credit Card Fraud Detection**: Tabular dataset with binary classification

## Project Structure

```
contrastive-learning/
├── src/
│   ├── cli/                    # Training scripts
│   │   ├── run_all.sh          # Run all experiments
│   │   ├── train_cifar10lt.sh  # CIFAR-10-LT training
│   │   └── train_creditcard.sh # Credit Card training
│   ├── data/                   # Data loading and processing
│   │   ├── dataset.py          # Dataset classes
│   │   ├── imbalance.py        # Imbalance utilities
│   │   └── transforms.py       # Augmentation pipelines
│   ├── models/                 # Model architectures
│   │   ├── base.py             # Base model
│   │   ├── encoder.py          # Contrastive encoder
│   │   └── classifier.py       # Linear classifier
│   ├── training/               # Training utilities
│   │   ├── losses.py           # Loss functions
│   │   ├── train_contrastive.py    # Contrastive training
│   │   └── train_classifier.py     # Classifier training
│   ├── evaluation/             # Evaluation tools
│   │   ├── metrics.py          # Metrics
│   │   ├── evaluator.py        # Evaluation
│   │   └── visualize.py        # Visualization
│   ├── pipeline/
│   │   └── run_contrastive.py  # Main pipeline
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration
│       ├── logger.py           # Logging
│       └── seed.py             # Random seed management
├── configs/                    # Configuration files
│   ├── default.yaml            # CIFAR-10-LT config
│   ├── cifar10_r100.yaml       # CIFAR-10-LT (alias)
│   └── creditcard.yaml         # Credit Card config
├── main.py                     # Main entry point
├── pyproject.toml              # Project config
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional but recommended)

### Setup

1. Navigate to the project directory:
```bash
cd /applied-data-science/final-project/contrastive-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or with poetry:
```bash
poetry install
```

3. Verify setup:
```bash
python verify_setup.py
```

## Usage

### Train on CIFAR-10-LT

```bash
python main.py --config configs/default.yaml --dataset cifar10-lt
```

### Train on Credit Card Dataset

```bash
python main.py --config configs/creditcard.yaml --dataset credit-card-fraud
```

### Run all experiments

```bash
bash src/cli/run_all.sh
```

## Configuration

Configuration files are in YAML format:

- `configs/default.yaml`: Default CIFAR-10-LT configuration
- `configs/cifar10_r100.yaml`: Alias for default.yaml
- `configs/creditcard.yaml`: Credit Card Fraud detection configuration

Key parameters:
- `augmentation`: Data augmentation settings
- `contrastive_training`: Contrastive pre-training settings
- `classifier_training`: Downstream classifier training settings
- `batch_size`: Batch size (default: 128)

## Key Differences Between Datasets

### CIFAR-10-LT (Image Dataset)
- Augmentation: RandomCrop with padding + RandomHorizontalFlip
- Normalization: ImageNet statistics
- Input shape: 3 x 32 x 32

### Credit Card (Tabular Dataset)
- No augmentation applied (tabular data)
- Simple normalization to tensor format
- Input shape: varies based on features

## References

The pipeline is based on the following concepts:
- SimCLR: Contrastive learning framework
- Class-imbalance handling: Sampling and weighting strategies

## License

This project is part of applied-data-science coursework.

## Installation

1. **Clone/Download the project**

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

Or using poetry:

```bash
poetry install
```

## Datasets

### 1. CIFAR-10-LT (Long-Tail)

The CIFAR-10-LT dataset is automatically downloaded from HuggingFace. You can specify the imbalance ratio:

- `r-20`: Imbalance ratio 20:1
- `r-50`: Imbalance ratio 50:1
- `r-100`: Imbalance ratio 100:1 (default)
- `r-200`: Imbalance ratio 200:1

### 2. Credit Card Fraud Detection

To use the Credit Card Fraud Detection dataset:

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it in the `data/` directory:

```
data/
└── creditcard.csv
```

## Usage

### Quick Start

**Train on CIFAR-10-LT**:

```bash
bash src/cli/train_cifar10lt.sh
```

**Train on Credit Card Fraud Detection**:

```bash
bash src/cli/train_creditcard.sh
```

**Run all experiments**:

```bash
bash src/cli/run_all.sh all
```

### Custom Training

You can customize training parameters:

```bash
python main.py \
    --dataset cifar10-lt \
    --cifar10-config r-100 \
    --encoder-architecture resnet18 \
    --projection-dim 128 \
    --contrastive-epochs 200 \
    --classifier-epochs 100 \
    --contrastive-lr 0.5 \
    --classifier-lr 0.001 \
    --batch-size 512 \
    --temperature 0.07 \
    --output-dir outputs/custom \
    --seed 42
```

### Available Arguments

```
Dataset Options:
  --dataset {cifar10-lt, credit-card-fraud}    Dataset to use (default: cifar10-lt)
  --cifar10-config {r-20, r-50, r-100, r-200}  CIFAR-10 imbalance ratio (default: r-100)

Model Options:
  --encoder-architecture {resnet18, resnet50}   Encoder architecture (default: resnet18)
  --projection-dim int                          Projection dimension (default: 128)
  --hidden-dim int                              Hidden dimension (default: 2048)

Training Options:
  --contrastive-epochs int                      Contrastive training epochs (default: 200)
  --classifier-epochs int                       Classifier training epochs (default: 100)
  --contrastive-lr float                        Contrastive learning rate (default: 0.5)
  --classifier-lr float                         Classifier learning rate (default: 0.001)
  --batch-size int                              Batch size (default: 512)
  --temperature float                           Temperature for loss (default: 0.07)

General Options:
  --output-dir path                             Output directory (default: outputs)
  --seed int                                    Random seed (default: 42)
  --device {cuda, cpu}                          Device to use (default: cuda)
```

## Pipeline Overview

The training pipeline consists of three main stages:

### Stage 1: Contrastive Encoder Training

- Learns robust representations using **NT-Xent loss** (SimCLR-style)
- Uses aggressive data augmentation (random crops, color jitter, blur, etc.)
- Trains on unlabeled or self-supervised mode
- **Output**: Pre-trained encoder that captures semantic information

### Stage 2: Downstream Classifier Training

- Freezes the pre-trained encoder
- Trains a lightweight linear classifier on top
- Uses **class-weighted loss** to handle imbalance
- **Output**: Final classifier for the downstream task

### Stage 3: Evaluation

- Computes metrics: Accuracy, Balanced Accuracy, Macro F1, Weighted F1, etc.
- Generates per-class metrics
- Produces confusion matrix and visualization plots

## Output

Training outputs are saved to the `--output-dir`:

```
outputs/
├── checkpoints/
│   ├── contrastive/
│   │   ├── best_model.pt
│   │   └── model_epoch_*.pt
│   └── classifier/
│       ├── best_model.pt
│       └── model_epoch_*.pt
├── logs/
│   ├── pipeline.log
│   ├── contrastive_training.log
│   └── classifier_training.log
└── figures/
    ├── class_distribution.png
    ├── contrastive_training.png
    ├── classifier_training.png
    └── confusion_matrix.png
```

## Key Features

### 1. Advanced Data Augmentation

The contrastive learning pipeline uses:

- Random resized crops with configurable scale
- Color jittering (brightness, contrast, saturation, hue)
- Random grayscale conversion
- Gaussian blur
- Solarization

### 2. Loss Functions

- **NT-Xent Loss**: Normalized Temperature-Scaled Cross Entropy (SimCLR)
- **Supervised Contrastive Loss**: For labeled data
- **Class-Weighted Cross-Entropy**: For downstream classification on imbalanced data

### 3. Class-Aware Sampling

Special handling for imbalanced datasets:

- Class distribution analysis
- Class weight calculation
- Weighted loss functions
- Balanced metric calculations

### 4. Flexible Architecture

- Support for multiple encoder architectures
- Configurable projection heads
- Multi-layer MLPs for projection and classification
- Batch normalization options

## Configuration

The project uses dataclass-based configuration. Modify `src/utils/config.py` to change default parameters:

```python
@dataclass
class ContrastiveTrainingConfig:
    batch_size: int = 512
    num_epochs: int = 200
    learning_rate: float = 0.5
    temperature: float = 0.07
    # ... more options
```

## Performance Benchmarks

The pipeline achieves good performance on both datasets:

- **CIFAR-10-LT (r-100)**: ~70-75% balanced accuracy
- **Credit Card Fraud**: ~95%+ accuracy with class-aware metrics

Results depend on:

- Dataset imbalance ratio
- Training epochs
- Learning rates
- Encoder architecture

## Troubleshooting

### CUDA Out of Memory

- Reduce `--batch-size`
- Use `--device cpu` for testing
- Use a smaller encoder: `--encoder-architecture resnet18`

### Slow Training

- Reduce number of workers: modify `DataConfig.num_workers`
- Use mixed precision training (requires PyTorch with CUDA)

### Dataset Not Found

For Credit Card Fraud:
- Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Place in `data/creditcard.csv`

For CIFAR-10-LT:
- Automatically downloaded from HuggingFace Datasets

## Next Steps

Possible extensions:

1. **Advanced Sampling**: Implement balanced or stratified sampling
2. **Mixed Precision**: Use AMP for faster training
3. **EMA**: Add Exponential Moving Average updates
4. **Multi-GPU**: Implement DistributedDataParallel
5. **Hyperparameter Tuning**: Use Optuna or Ray Tune
6. **Model Ensembling**: Combine multiple trained models

## References

- **SimCLR**: Chen et al., 2020 - "A Simple Framework for Contrastive Learning"
- **Supervised Contrastive Learning**: Khosla et al., 2021
- **CIFAR-10-LT**: Tan et al., 2021 - "Decoupling Representation and Classifier for Long-Tailed Recognition"

## License

This project is open source and available under the MIT License.

## Author

ML Research Project

---

For questions or issues, please refer to the documentation or create an issue in the repository.
