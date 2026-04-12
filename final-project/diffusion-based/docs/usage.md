# Usage Guide

## Installation

### Option 1: Conda + Pip

```bash
# Create conda environment
conda create -n data-imbalanced python=3.10
conda activate data-imbalanced

# Install dependencies
pip install -e .
```

### Option 2: Poetry

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install with poetry
poetry install
poetry shell
```

### Option 3: Docker (if available)

```bash
docker build -t diffusion-imbalance .
docker run --gpus all -it diffusion-imbalance
```

## Quick Start

### 1. Run Baseline (No Balancing)

```bash
# Navigate to project
cd diffusion-based

# Run baseline pipeline
python -m src.pipeline.run_baseline

# Or use script
cd src/cli
bash run_baseline.sh
```

**Output**:
- Logs: `outputs/logs/baseline/`
- Checkpoints: `outputs/checkpoints/baseline/`
- Figures: `outputs/figures/baseline/`

### 2. Run Diffusion Pipeline

```bash
# Full pipeline (diffusion + classifier)
python -m src.pipeline.run_diffusion_pipeline

# Or use script
cd src/cli
bash run_diffusion.sh
```

**Output**:
- Logs: `outputs/logs/diffusion/`
- Checkpoints: `outputs/checkpoints/diffusion/`
- Figures: `outputs/figures/diffusion/`

### 3. Compare Results

Results are automatically saved with logs and figures. Compare manually:

```bash
# View logs
tail -50 outputs/logs/baseline/classifier.log
tail -50 outputs/logs/diffusion/classifier.log

# Check figures
ls outputs/figures/baseline/
ls outputs/figures/diffusion/
```

## Configuration

### Via YAML Files

Edit configuration files before running:

```yaml
# configs/diffusion.yaml
diffusion:
  num_timesteps: 1000      # Increase for better quality
  num_epochs: 100          # Training duration
  batch_size: 64           # Reduce if OOM
  num_samples_per_class: 500  # More synthetic data

# configs/classifier.yaml
classifier:
  num_epochs: 200          # Training duration
  learning_rate: 0.001     # Adjust based on convergence
  batch_size: 128          # Increase if memory available
```

### Via Python Code

```python
from src.utils.config import DiffusionConfig, ClassifierConfig
from src.pipeline.run_diffusion_pipeline import run_diffusion_pipeline

# Create custom configs
diff_cfg = DiffusionConfig(
    num_timesteps=1000,
    num_epochs=50,      # Shorter training for testing
    batch_size=32,      # Smaller for resource limits
)

clf_cfg = ClassifierConfig(
    learning_rate=0.0005,
    num_epochs=100,
)

# Run pipeline
metrics = run_diffusion_pipeline(
    diffusion_config=diff_cfg,
    classifier_config=clf_cfg,
)

print(f"Final Accuracy: {metrics['accuracy']:.4f}")
```

## Advanced Usage

### Custom Data Pipeline

```python
from src.data.dataset import CIFAR10LTDataset, create_data_loaders
from src.data.transforms import get_train_transforms, get_test_transforms

# Load custom split
dataset = CIFAR10LTDataset(
    split="train",
    transform=get_train_transforms(image_size=32),
    dataset_name="tomas-gajarsky/cifar10-lt",
    dataset_config="r-20",
)

# Create dataloader
loader = create_data_loaders(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
)

# Analyze imbalance
from src.data.imbalance import analyze_class_distribution, print_class_distribution
dist = analyze_class_distribution([dataset.labels[i] for i in range(len(dataset))])
print_class_distribution(dist)
```

### Custom Model Training

```python
from pathlib import Path
from src.models.diffusion import DiffusionModel
from src.training.train_diffusion import train_diffusion
from src.utils.seed import set_seed

set_seed(42)

# Create model
model = DiffusionModel(
    num_timesteps=1000,
    num_classes=10,
    model_channels=64,
    num_residual_blocks=2,
    use_ema=True,
)

# Train
history = train_diffusion(
    model=model,
    train_loader=train_loader,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda",
    checkpoint_dir=Path("outputs/checkpoints/custom"),
    log_dir=Path("outputs/logs/custom"),
)

print(f"Final loss: {history['loss'][-1]:.4f}")
```

### Custom Sampling

```python
from src.pipeline.sampling import sample_from_diffusion, sample_from_diffusion_fast

# Full quality (slow)
images, labels = sample_from_diffusion(
    model=diffusion_model,
    num_samples_per_class=500,
    image_size=32,
)

# Fast sampling (less steps)
images_fast, labels_fast = sample_from_diffusion_fast(
    model=diffusion_model,
    num_samples_per_class=500,
    num_steps=100,  # 10x faster
)

print(f"Generated {len(images)} images")
```

### Custom Evaluation

```python
from src.evaluation.evaluator import evaluate_classifier
from src.evaluation.visualize import plot_confusion_matrix, plot_training_curves

# Evaluate model
metrics, per_class = evaluate_classifier(
    model=classifier,
    data_loader=test_loader,
    num_classes=10,
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

# Plot results
plot_confusion_matrix(
    cm=metrics_calc.get_confusion_matrix(),
    save_path=Path("outputs/figures/custom_confusion.png"),
)

plot_training_curves(
    history=train_history,
    save_path=Path("outputs/figures/custom_training.png"),
)
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Reduce batch size
```yaml
# configs/diffusion.yaml
diffusion:
  batch_size: 32  # Reduce from 64
```

**Solution 2**: Reduce model size
```yaml
diffusion:
  model_channels: 32  # Reduce from 64
```

**Solution 3**: Use gradient accumulation
```python
# In training loop:
# Accumulate gradients over multiple steps before step()
```

### Issue: "Dataset download fails"

**Solution**: Pre-download dataset
```python
from datasets import load_dataset
dataset = load_dataset("tomas-gajarsky/cifar10-lt", "r-20")
# Cache will be saved for future use
```

### Issue: "GPU not detected"

**Solution**: Check CUDA installation
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, set to CPU
# Edit configs or code:
device = "cpu"  # Instead of "cuda"
```

### Issue: "Training loss not decreasing"

**Causes & Solutions**:
1. Learning rate too high → Reduce LR by 10x
2. Batch size too small → Increase batch size
3. Model too small → Increase model_channels
4. Need more epochs → Increase num_epochs

### Issue: "Generated samples look like noise"

**Causes & Solutions**:
1. Diffusion not trained long enough → Increase epochs
2. Learning rate too high → Reduce LR
3. Batch size too high → Reduce batch size
4. Using untrained EMA model → Check EMA settings

## Performance Tuning

### For Faster Execution

```yaml
# configs/diffusion.yaml
diffusion:
  num_timesteps: 500      # Fewer steps (less accurate)
  batch_size: 128         # Larger batches if memory allows
  num_epochs: 50          # Fewer epochs for testing
  model_channels: 32      # Smaller model
```

```python
# Use fast sampling
images, labels = sample_from_diffusion_fast(
    model=model,
    num_steps=100,  # 10x faster than 1000
)
```

### For Better Quality

```yaml
diffusion:
  num_timesteps: 2000     # More steps
  beta_schedule: "cosine" # Better variance schedule
  batch_size: 32          # Smaller for better stats
  num_epochs: 200         # Train longer
  num_samples_per_class: 1000  # More synthetic data
```

## Monitoring Training

### Real-Time Monitoring

Training logs are printed to console and saved to files:

```bash
# Watch logs in real-time
tail -f outputs/logs/diffusion/diffusion.log

# Count training steps
grep "Step" outputs/logs/diffusion/diffusion.log | wc -l
```

### TensorBoard (Optional)

```bash
# If you add TensorBoard support to logging:
tensorboard --logdir outputs/logs/

# View at http://localhost:6006
```

## Saving & Loading Models

### Save Checkpoint

```python
import torch

# During training, checkpoints are auto-saved
# Manual save:
torch.save(model.state_dict(), "outputs/checkpoints/custom/model.pt")
```

### Load Checkpoint

```python
import torch
from src.models.diffusion import DiffusionModel

# Create model
model = DiffusionModel()

# Load weights
checkpoint = torch.load("outputs/checkpoints/diffusion/diffusion_epoch_100.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Ready to use
model.eval()
```

## Generating New Samples

```python
from src.models.diffusion import DiffusionModel
from src.pipeline.sampling import sample_from_diffusion
import torch

# Load trained model
model = DiffusionModel()
checkpoint = torch.load("path/to/checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate samples
images, labels = sample_from_diffusion(
    model=model,
    num_samples_per_class=100,  # 100 per class = 1000 total
    image_size=32,
    device="cuda",
)

print(f"Generated {len(images)} samples")

# Save samples (optional)
from torchvision.transforms.functional import to_pil_image
from pathlib import Path

save_dir = Path("outputs/generated_samples")
save_dir.mkdir(exist_ok=True)

for idx, (img, label) in enumerate(zip(images, labels)):
    # Denormalize if needed
    pil_img = to_pil_image(img.clamp(0, 1))
    pil_img.save(save_dir / f"class_{label}_sample_{idx}.png")
```

## Best Practices

### For Research

1. **Document Everything**: Modify configs, note changes
2. **Use Version Control**: Track experiments with git
3. **Save Artifacts**: Checkpoints, logs, generated samples
4. **Reproduce Baselines**: Always trained baseline for comparison
5. **Statistical Significance**: Run multiple seeds (42, 123, 456)

### For Production

1. **Optimize Inference**: Use DDIM for speed
2. **Batch Processing**: Process multiple samples in parallel
3. **Monitor Resources**: Track memory, CPU usage
4. **Error Handling**: Validate inputs, catch exceptions
5. **Logging**: Comprehensive logging for debugging

### For Debugging

1. **Start Small**: Test with subset of data first
2. **Use Breakpoints**: Insert pdb.set_trace() for inspection
3. **Log Intermediate**: Print shapes, values at each step
4. **Visualize**: Plot generated samples regularly
5. **Compare**: Baseline vs. new approach
