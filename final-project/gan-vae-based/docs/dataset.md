# Dataset Documentation

## CIFAR-10-LT Overview

**CIFAR-10-LT** is a long-tailed (imbalanced) version of CIFAR-10 where class frequencies follow a power-law distribution.

### Dataset Statistics

- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Samples**: ~12,500 (imbalanced)
- **Test Samples**: 10,000 (balanced)
- **Image Size**: 32×32 pixels
- **Channels**: RGB (3 channels)

### Imbalance Ratios

The dataset is available with different imbalance ratios (r):

- **r-10**: Imbalance ratio 10:1 (majority:minority class)
- **r-20**: Imbalance ratio 20:1 (used in this project)
- **r-50**: Imbalance ratio 50:1 (more challenging)
- **r-100**: Imbalance ratio 100:1 (highly imbalanced)

### CIFAR-10-LT (r-20) Distribution

For r-20, the approximate per-class distribution is:

```
Class 0 (airplane):   500 samples
Class 1 (automobile): 500 samples
Class 2 (bird):       370 samples
Class 3 (cat):        270 samples
Class 4 (deer):       195 samples
Class 5 (dog):        150 samples
Class 6 (frog):       110 samples
Class 7 (horse):      80 samples
Class 8 (ship):       60 samples
Class 9 (truck):      45 samples
```

Total: ~12,500 training samples

### Why Imbalanced Datasets Are Challenging

1. **Naive Accuracy**: Achieves high accuracy by always predicting majority class
2. **Minority Neglect**: Model underfits minority classes
3. **Class-wise Performance Varies**: Minority classes get poor recalls
4. **Evaluation Misleading**: Standard accuracy != real performance
5. **Decision Boundaries**: Minority class boundaries pushed away

## Loading the Dataset

### From HuggingFace Hub

```python
from datasets import load_dataset

# Load CIFAR-10-LT with r-20 imbalance
ds = load_dataset("tomas-gajarsky/cifar10-lt", "r-20")

# Access splits
train_data = ds['train']
test_data = ds['test']

# Each sample has: {'image': PIL.Image, 'label': int}
sample = train_data[0]
print(sample.keys())  # dict_keys(['image', 'label'])
```

### In This Project

```python
from src.data.dataset import get_train_val_datasets

# Automatically downloads and preprocesses
train_ds, val_ds, test_ds = get_train_val_datasets(
    dataset_name="tomas-gajarsky/cifar10-lt",
    config_name="r-20",
    val_split=0.1,  # 10% of train for validation
    image_size=32,
)
```

## Data Preprocessing

### Normalization

CIFAR-10-LT uses ImageNet normalization statistics:

```
Mean: (0.4914, 0.4822, 0.4465)
Std:  (0.2023, 0.1994, 0.2010)
```

### Training Transforms

```python
transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=..., std=...),
])
```

### Test/Validation Transforms

```python
transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=..., std=...),
])
```

## Class Distribution Analysis

### Identification of Minority Classes

In this project, minority classes are identified as those below the 50th percentile:

```python
from src.data.imbalance import identify_minority_classes

minority = identify_minority_classes(targets, threshold=0.5)
# Returns: [6, 7, 8, 9] for r-20
```

### Imbalance Ratio Calculation

```python
from src.data.imbalance import get_imbalance_ratio

ratio = get_imbalance_ratio(targets)
# For r-20 CIFAR-10-LT: ~11 (500 / 45)
```

## Oversampling Analysis

### Samples Needed Calculation

```python
from src.data.imbalance import get_samples_needed

# To achieve balanced distribution (1.0 ratio)
needed = get_samples_needed(targets, target_ratio=1.0)
# Returns: {6: 1040, 7: 1070, 8: 1090, 9: 1105}
# (samples to generate per minority class)
```

### Class Distribution Evolution

**Original (r-20):**
- Majority: 500 samples (airplane, automobile)
- Minority: 45 samples (truck)
- Ratio: 11:1

**After Oversampling (Balanced):**
- All classes: 500 samples each
- Ratio: 1:1
- Total samples: 5000 (from original 12,500)

## Data Characteristics for Generative Models

### Why CIFAR-10 is Suitable for GAN/VAE

1. **Small Images**: 32×32 is easy to generate (compared to ImageNet)
2. **Well-Defined Classes**: Clear visual patterns for each class
3. **RGB Data**: Standard image generation without special handling
4. **Class-Conditional**: Each image has clear label for conditional generation
5. **Challenging Distribution**: Imbalance tests generalization

### What Makes Minority Classes Hard

Minority classes like "truck" have:

- **Limited Training Examples**: Only 45 samples vs 500 for majority
- **High Feature Diversity**: All variants must be learned from few examples
- **Long-Tail Characteristics**: Rare combinations may be underrepresented
- **Classifier Bias**: Trained on imbalanced data, prefers majority classes

## Data Quality Checks

Run these to inspect data:

```python
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

loader = DataLoader(train_ds, batch_size=16)
images, labels = next(iter(loader))

# Visualize
fig, axes = plt.subplots(4, 4)
for ax, img, label in zip(axes.flat, images, labels):
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(f"Class {label}")
    ax.axis('off')
```

## Dataset Variants

To experiment with different imbalance levels:

```python
# More imbalanced
ds_r50 = load_dataset("tomas-gajarsky/cifar10-lt", "r-50")

# Less imbalanced
ds_r10 = load_dataset("tomas-gajarsky/cifar10-lt", "r-10")
```

## Performance Expectations

**On Balanced Test Set:**

- **Baseline Classifier**: ~75% accuracy (but poor minority recall)
- **After GAN Oversampling**: ~82% accuracy
- **After VAE Oversampling**: ~80% accuracy

**Per-Class Recall (Minority):**

- **Baseline**: ~30-40% recall on minority classes
- **After Oversampling**: ~70-80% recall on minority classes

## References

- Original Paper: [He et al., 2020](https://arxiv.org/abs/2007.07314)
- Dataset Hub: [HuggingFace Datasets](https://huggingface.co/datasets/tomas-gajarsky/cifar10-lt)
- CIFAR-10: [Krizhevsky et al., 2009](https://www.cs.toronto.edu/~kriz/cifar.html)
