# CIFAR-10-LT Dataset

## Overview

CIFAR-10-LT is a long-tail imbalanced version of the original CIFAR-10 dataset. It simulates realistic class imbalance found in real-world applications.

## Dataset Details

### Original CIFAR-10
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Samples**: 50,000 (5,000 per class)
- **Test Samples**: 10,000 (1,000 per class)
- **Image Size**: 32×32 RGB
- **Format**: Numpy arrays

### CIFAR-10-LT (Long-Tail)

Creates imbalanced distribution:

#### r-20 Configuration (Used in this project)
- **Imbalance Ratio**: 20 (max/min class samples)
- **Head Classes**: Many samples (e.g., 500)
- **Tail Classes**: Few samples (e.g., 25)
- **Total Training Samples**: ~12,500 (reduced from 50,000)

Example distribution:
```
Class 0 (airplane):  500 samples
Class 1 (automobile): 450 samples
...
Class 8 (ship):      50 samples
Class 9 (truck):     25 samples
```

## Problem Definition

The class imbalance creates challenges:

1. **Bias**: Model biases toward head classes
2. **Poor Minority Performance**: Tail classes have low recall
3. **Loss Dominance**: Head class loss dominates training
4. **Overfitting**: Risk of memorizing tail classes

## Loading the Dataset

### Using HuggingFace

```python
from datasets import load_dataset

# Load CIFAR-10-LT r-20
ds = load_dataset("tomas-gajarsky/cifar10-lt", "r-20")

# Access splits
train_split = ds["train"]
test_split = ds["test"]

# Sample structure
sample = train_split[0]
# {'img': PIL.Image, 'label': int}
```

### Using This Project

```python
from src.data.dataset import CIFAR10LTDataset

# Create dataset
dataset = CIFAR10LTDataset(
    split="train",
    dataset_name="tomas-gajarsky/cifar10-lt",
    dataset_config="r-20",
)

# Analyze distribution
from src.data.imbalance import analyze_class_distribution
dist = analyze_class_distribution([dataset.labels[i] for i in range(len(dataset))])
```

## Data Characteristics

### Class Distribution (r-20)

| Class | Samples | Percentage |
|-------|---------|-----------|
| 0     | 500     | 4.0%     |
| 1     | 450     | 3.6%     |
| 2     | 405     | 3.2%     |
| ...   | ...     | ...      |
| 8     | 50      | 0.4%     |
| 9     | 25      | 0.2%     |
|Total  | 12,500  | 100%     |

### Visual Characteristics

- **Well-Balanced Classes**: Balanced lighting, angles, backgrounds
- **Natural Variation**: Real-world object appearance variation
- **Fixed Size**: All images 32×32 (appropriate for diffusion models)
- **Color Images**: Full RGB color information

## Data Preprocessing

### Normalization Parameters (CIFAR-10)

Standard ImageNet-style normalization:

```python
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
```

### Augmentation Strategy

#### Training
- Random crop (32x32 with 4px padding)
- Random horizontal flip (p=0.5)
- Normalization

#### Testing
- No augmentation
- Center crop to 32x32
- Normalization

### No-Normalization Version (Diffusion)

Diffusion models operate on unnormalized images [0, 1]:

```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # Converts to [0, 1]
])
```

## Class Statistics

### Long-Tail Distribution Properties

1. **Exponential Decay**: Samples decrease exponentially
2. **Few Tail Samples**: Minority classes critical but rare
3. **High Variance**: Tail classes have high sample variance
4. **Real-World Relevance**: Simulates production class imbalance

### Challenges for Models

| Aspect | Challenge | Impact |
|--------|-----------|--------|
| Loss | Head loss dominates | Model ignores tail |
| Metrics | Accuracy misleading | Macro-F1 more important |
| Sampling | Biased sampling | Need to balance batches |
| Learning | Different rates | Tail classes learn slower |

## Handling Strategy

### This Project's Approach

1. **Diffusion-Based Oversampling**
   - Generate synthetic minority samples
   - Learn class-conditional generation
   - Preserve feature distributions

2. **Dataset Balancing**
   - Oversample minority classes
   - Undersample majority classes (optional)
   - Create balanced training set

3. **Improved Training**
   - Train on balanced dataset
   - Better class representation
   - Improved generalization

## Evaluation Metrics

Account for imbalance with:

- **Macro F1**: Average across classes (equal weight)
- **Per-Class Recall**: Individual class performance
- **Weighted F1**: Class-weighted average
- **Confusion Matrix**: Detailed class interaction

## Related Work

### Dataset References
- Original: Krizhevsky (2009) CIFAR-10
- Long-tail: Cao et al. (2019) "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
- HuggingFace: Community maintained version

### Solutions to Class Imbalance
1. **Resampling**: Over/undersampling
2. **Reweighting**: Loss function weights
3. **Transfer Learning**: Pre-trained models
4. **Data Augmentation**: Synthetic sample generation
5. **Ensemble Methods**: Multiple models

## Downloads & Caching

### First Run
- Automatic download (~150MB)
- Cached in `~/.cache/huggingface/datasets/`
- Subsequent runs use cache

### Manual Download
```bash
python -c "from datasets import load_dataset; load_dataset('tomas-gajarsky/cifar10-lt', 'r-20')"
```

## Data Splits

This project uses:

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | ~12,500 | Diffusion training; Classifier training |
| Test  | 10,000  | Classifier evaluation (balanced) |

The test set is balanced, making metrics more comparable across classes.
