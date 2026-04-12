# Diffusion-Based Oversampling for Imbalanced CIFAR-10-LT

A complete, production-quality Python project demonstrating handling imbalanced data using diffusion-based oversampling on CIFAR-10-LT.

## Overview

This project implements a comprehensive pipeline for addressing class imbalance in the CIFAR-10-LT dataset using conditional Denoising Diffusion Probabilistic Models (DDPM). The approach generates synthetic samples for minority classes, creating a balanced dataset for improved classifier training.

### Key Features

- **Conditional Diffusion Model**: Class-conditioned DDPM with U-Net architecture
- **Synthetic Data Generation**: Generate balanced synthetic samples for minority classes
- **End-to-End Pipeline**: Complete workflow from data loading to evaluation
- **Comprehensive Evaluation**: Metrics including accuracy, F1-score, and per-class recall
- **Reproducibility**: Fixed random seeds and full type hints
- **Logging & Visualization**: Detailed logs and publication-quality figures

## Installation

### Prerequisites

- Python 3.10+
- Conda or Mamba (recommended)
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup using Conda

```bash
# Create conda environment
conda create -n data-imbalanced python=3.10
conda activate data-imbalanced

# Install dependencies
pip install -e .
```

### Setup using Poetry

```bash
# Install poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

## Quick Start

### Run Baseline (Original Imbalanced Data)

```bash
cd src/cli
bash run_baseline.sh
```

This trains a classifier on the original CIFAR-10-LT without any balancing.

### Run Diffusion Pipeline

```bash
cd src/cli
bash run_diffusion.sh
```

This runs the complete pipeline:
1. Trains diffusion model on original data
2. Generates synthetic samples for minority classes
3. Trains classifier on balanced dataset
4. Evaluates and compares results

### Run Specific Component

```bash
# Train diffusion model only
python -m src.training.train_diffusion

# Train classifier only
python -m src.training.train_classifier

# Evaluate classifier
python -m src.evaluation.evaluator
```

## Project Structure

```
project_root/
├── src/
│   ├── data/
│   │   ├── dataset.py          # CIFAR-10-LT dataset loading
│   │   ├── transforms.py       # Data augmentation
│   │   └── imbalance.py        # Imbalance analysis
│   │
│   ├── models/
│   │   ├── base.py             # Base model class and EMA
│   │   ├── diffusion.py        # DDPM implementation
│   │   ├── unet.py             # U-Net architecture
│   │   ├── scheduler.py        # Noise scheduler
│   │   └── classifier.py       # Classification models
│   │
│   ├── training/
│   │   ├── train_diffusion.py  # Diffusion training loop
│   │   ├── train_classifier.py # Classifier training loop
│   │   └── losses.py           # Loss functions
│   │
│   ├── evaluation/
│   │   ├── evaluator.py        # Evaluation utilities
│   │   ├── metrics.py          # Metric calculations
│   │   └── visualize.py        # Visualization functions
│   │
│   ├── pipeline/
│   │   ├── run_baseline.py     # Baseline pipeline
│   │   ├── run_diffusion_pipeline.py  # Diffusion pipeline
│   │   └── sampling.py         # Sampling utilities
│   │
│   ├── utils/
│   │   ├── config.py           # Configuration management
│   │   ├── logger.py           # Logging utilities
│   │   └── seed.py             # Reproducibility
│   │
│   └── cli/
│       ├── run_baseline.sh            # Baseline script
│       ├── run_diffusion.sh           # Diffusion script
│       └── evaluate.sh                # Evaluation script
│
├── configs/
│   ├── diffusion.yaml          # Diffusion configuration
│   └── classifier.yaml         # Classifier configuration
│
├── outputs/
│   ├── logs/                   # Training logs
│   ├── checkpoints/            # Model checkpoints
│   └── figures/                # Visualizations
│
├── docs/
│   ├── overview.md             # Project overview
│   ├── dataset.md              # Dataset description
│   ├── methods.md              # Method explanation
│   ├── experiments.md          # Experimental setup
│   └── usage.md                # Usage guide
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

Configuration is managed via:

1. **Dataclasses** in `src/utils/config.py`
2. **YAML files** in `configs/` directory

### Diffusion Configuration (`configs/diffusion.yaml`)

```yaml
diffusion:
  num_timesteps: 1000        # Number of diffusion steps
  beta_start: 0.0001         # Starting variance
  beta_end: 0.02             # Ending variance
  beta_schedule: "linear"    # Schedule: linear or cosine
  model_channels: 64         # Base channel count
  num_residual_blocks: 2     # Residual blocks per level
  num_samples_per_class: 500 # Synthetic samples per class
```

### Classifier Configuration (`configs/classifier.yaml`)

```yaml
classifier:
  model_name: "resnet18"   # Model: resnet18 or lightweight
  learning_rate: 0.001     # Learning rate
  batch_size: 128          # Batch size
  num_epochs: 200          # Training epochs
```

## Usage Examples

### Basic Pipeline

```python
from pathlib import Path
from src.pipeline.run_diffusion_pipeline import run_diffusion_pipeline
from src.utils.config import DiffusionConfig, ClassifierConfig

# Run with custom config
diffusion_cfg = DiffusionConfig(num_epochs=50)
classifier_cfg = ClassifierConfig(num_epochs=100)

metrics = run_diffusion_pipeline(
    output_dir=Path("outputs"),
    diffusion_config=diffusion_cfg,
    classifier_config=classifier_cfg,
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

### Custom Training

```python
import torch
from src.models.diffusion import DiffusionModel
from src.training.train_diffusion import train_diffusion

# Create model
model = DiffusionModel(
    num_timesteps=1000,
    num_classes=10,
    model_channels=64,
)

# Train
history = train_diffusion(
    model=model,
    train_loader=train_loader,
    num_epochs=100,
    learning_rate=1e-4,
)
```

### Sampling

```python
from src.pipeline.sampling import sample_from_diffusion

# Generate synthetic data
synthetic_images, synthetic_labels = sample_from_diffusion(
    model=diffusion_model,
    num_samples_per_class=500,
)
```

## Output Structure

After running pipelines, outputs are organized as:

```
outputs/
├── logs/
│   ├── baseline/              # Baseline training logs
│   │   ├── baseline.log
│   │   ├── classifier.log
│   │   └── ...
│   │
│   └── diffusion/             # Diffusion pipeline logs
│       ├── diffusion_pipeline.log
│       ├── diffusion.log
│       ├── classifier.log
│       └── ...
│
├── checkpoints/
│   ├── baseline/              # Baseline model checkpoints
│   │   ├── classifier_best.pt
│   │   └── classifier_epoch_*.pt
│   │
│   └── diffusion/             # Diffusion model checkpoints
│       ├── diffusion_epoch_*.pt
│       └── classifier_best.pt
│
└── figures/
    ├── baseline/              # Baseline visualizations
    │   └── training_curves.png
    │
    └── diffusion/             # Diffusion visualizations
        ├── original_distribution.png
        ├── balanced_distribution.png
        ├── training_curves.png
        ├── diffusion_training.png
        ├── classifier_training.png
        └── generated_samples.png
```

## Key Metrics

The project evaluates using:

- **Accuracy**: Overall classification accuracy
- **Macro F1-Score**: Average F1 across all classes (equal weight)
- **Weighted F1-Score**: F1 weighted by class support
- **Per-Class Recall**: Individual class performance
- **Confusion Matrix**: Class-wise predictions

## Implementation Details

### Diffusion Model

The implementation follows the DDPM framework:

1. **Forward Process**: Gradually add Gaussian noise
   ```
   q(x_t|x_0) = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε
   ```

2. **Reverse Process**: Learn to denoise step-by-step
   ```
   p_θ(x_{t-1}|x_t) trained with MSE loss
   ```

3. **Conditioning**: Class labels embedded and injected

### U-Net Architecture

- **Input**: Noisy image (32x32x3) + timestep + class
- **Backbone**: Residual blocks with attention
- **Output**: Predicted noise (32x32x3)

### Class Balancing

1. Identify minority classes
2. Train diffusion model on original distribution
3. Sample from minority classes until balanced
4. Combine synthetic + real for training

## Performance Tips

### For Faster Training

1. Reduce `num_timesteps` (e.g., 500)
2. Use smaller `model_channels` (e.g., 32)
3. Reduce `num_epochs`
4. Use `lightweight` classifier model
5. Reduce `num_workers` if on limited resources

### For Better Results

1. Increase `num_timesteps` (e.g., 2000)
2. Use cosine `beta_schedule`
3. Increase training epochs
4. Use EMA (enabled by default)
5. Reduce learning rate with warmup

### GPU Memory

- Default config: ~8GB VRAM
- Reduce `batch_size` if out of memory
- Use gradient accumulation for effective larger batches

## Reproducibility

All runs use fixed random seeds controlled via:

```python
from src.utils.seed import set_seed
set_seed(42)  # Set seed to 42
```

Seeds are set for:
- Python `random`
- NumPy
- PyTorch (CPU and CUDA)
- CuDNN algorithms

## Dependencies

Core dependencies:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `datasets>=2.14.0`: For CIFAR-10-LT loading
- `pyyaml>=6.0`: Configuration management
- `scikit-learn>=1.3.0`: Metrics
- `matplotlib>=3.7.0`: Visualization
- `seaborn>=0.12.0`: Enhanced plots

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in config
sed -i 's/batch_size: 64/batch_size: 32/' configs/diffusion.yaml

# Or reduce model channels
sed -i 's/model_channels: 64/model_channels: 32/' configs/diffusion.yaml
```

### Dataset Download Issues

```bash
# The dataset is downloaded automatically on first run
# If issues persist, manually download:
python -c "from datasets import load_dataset; load_dataset('tomas-gajarsky/cifar10-lt', 'r-20')"
```

### GPU Not Detected

```bash
# Set device to CPU
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU (slower):
# Modify training scripts to use device='cpu'
```