# Usage Guide

## Getting Started

### 1. Installation

```bash
cd /Users/tgng_mac/Coding/applied-data-science/final-project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### 2. Verify Dataset Access

```bash
python -c "from datasets import load_dataset; ds = load_dataset('tomas-gajarsky/cifar10-lt', 'r-20'); print(ds)"
```

## Running Pipelines

### Quick Start (All 3 Methods)

```bash
# Run all pipelines
bash src/cli/run_all.sh

# Results appear in outputs/
ls -la outputs/
# outputs/
# ├── baseline/
# ├── gan/
# └── vae/
```

### Method 1: Baseline (No Oversampling)

```bash
# Run baseline
python -m src.cli.run_baseline --config configs/default.yaml

# Or with short hand
python -m src.cli.run_baseline

# Results
outputs/
└── baseline/
    ├── logs/
    │   └── __main__.log
    ├── checkpoints/
    │   └── classifier_best.pt
    ├── figures/
    │   ├── class_distribution_baseline.png
    │   └── training_curves_baseline.png
    ├── test_metrics.txt
    └── config.yaml
```

### Method 2: GAN-Based Oversampling

```bash
# Run GAN pipeline
python -m src.cli.run_gan --config configs/gan.yaml

# Detailed output
# Training GAN... (30 epochs)
# Generating synthetic samples... (samples needed per minority class)
# Training classifier... (100 epochs on balanced data)
# Evaluating...

# Results
outputs/
└── gan/
    ├── logs/
    │   └── __main__.log
    ├── checkpoints/
    │   ├── gan_epoch_*.pt
    │   └── classifier_best.pt
    ├── figures/
    │   ├── class_distribution_original.png
    │   ├── generated_images_gan.png
    │   ├── distribution_comparison_gan.png
    │   └── training_curves_classifier_gan.png
    ├── test_metrics.txt
    └── config.yaml
```

### Method 3: VAE-Based Oversampling

```bash
# Run VAE pipeline
python -m src.cli.run_vae --config configs/vae.yaml

# Similar structure to GAN but with VAE results
outputs/
└── vae/
    ├── logs/
    ├── checkpoints/
    ├── figures/
    │   ├── generated_images_vae.png  # Smoother, blurrier than GAN
    │   └── ...
    ├── test_metrics.txt
    └── config.yaml
```

## Customization

### Modify Configuration (YAML)

```bash
# Edit config before running
nano configs/gan.yaml

# Change batch size
batch_size: 16  # from 32

# Change epochs
epochs: 20      # from 50

# Change learning rate
learning_rate: 0.0001  # from 0.0002

# Save and run
python -m src.cli.run_gan --config configs/gan.yaml
```

### Override via Command-Line

```bash
# Change output directory
python -m src.cli.run_gan --output-dir /custom/path

# Results appear in /custom/path/gan/
```

## Interpreting Results

### Checking Metrics

```bash
# View results as text
cat outputs/gan/test_metrics.txt

# Output:
# Evaluation Metrics
# ==================================================
# Overall Metrics:
#   Accuracy: 0.8234
#   Balanced Accuracy: 0.7512
#   Macro F1: 0.7401
#   Macro Precision: 0.7389
#   Macro Recall: 0.7512
#
# Per-class Recall:
#   Class 0: 0.9234  (majority class - airplane)
#   Class 1: 0.8934  (majority class - automobile)
#   ...
#   Class 8: 0.6234  (minority class - ship)
#   Class 9: 0.5934  (minority class - truck)
```

### Viewing Visualizations

```bash
# View generated samples
open outputs/gan/figures/generated_images_gan.png

# View class distribution before/after
open outputs/gan/figures/distribution_comparison_gan.png

# View training curves
open outputs/gan/figures/training_curves_classifier_gan.png
```

### Comparing Methods

```bash
# Side-by-side comparison
cat outputs/baseline/test_metrics.txt
cat outputs/gan/test_metrics.txt
cat outputs/vae/test_metrics.txt

# Expected pattern:
# Baseline:  accuracy=0.75, balanced=0.55, f1=0.52
# GAN:       accuracy=0.82, balanced=0.75, f1=0.74
# VAE:       accuracy=0.80, balanced=0.72, f1=0.71
```

## Advanced Usage

### Evaluating Existing Model

```python
import torch
from src.evaluation.evaluator import Evaluator
from src.data.dataset import get_train_val_datasets
from torch.utils.data import DataLoader

# Load pretrained model
model = torch.load("outputs/gan/checkpoints/classifier_best.pt")
model.eval()

# Get test data
_, _, test_ds = get_train_val_datasets()
test_loader = DataLoader(test_ds, batch_size=32)

# Evaluate
evaluator = Evaluator(model, torch.device("cuda"), num_classes=10)
metrics = evaluator.evaluate(test_loader)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['macro_f1']:.4f}")
```

### Training Custom GAN

```python
from pathlib import Path
from src.utils.config import PipelineConfig
from src.models.gan import ConditionalGenerator, ConditionalDiscriminator
from src.training.train_gan import GANTrainer
from torch.utils.data import DataLoader
import torch

# Custom config
config = PipelineConfig()
config.gan.epochs = 100
config.gan.learning_rate = 0.0001

# Create models
device = torch.device("cuda")
generator = ConditionalGenerator(
    latent_dim=config.gan.latent_dim,
    num_classes=10,
    hidden_dim=config.gan.generator_hidden_dim,
)
discriminator = ConditionalDiscriminator(
    num_classes=10,
    hidden_dim=config.gan.discriminator_hidden_dim,
)

# Train
trainer = GANTrainer(
    generator, discriminator,
    vars(config.gan),
    device,
    checkpoint_dir=Path("custom_checkpoints")
)

# Your data loader here
# trainer.fit(train_loader)
```

### Generating Samples Only

```python
import torch
from src.models.gan import ConditionalGenerator

# Load pretrained generator
device = torch.device("cuda")
generator = ConditionalGenerator()
state = torch.load("outputs/gan/checkpoints/gan_epoch_29.pt")
generator.load_state_dict(state['generator_state'])
generator.eval().to(device)

# Generate samples of class 9 (truck - minority)
with torch.no_grad():
    z = torch.randn(100, 100).to(device)  # 100 samples, 100-D noise
    y = torch.full((100,), 9).to(device)  # class 9
    samples = generator(z, y)  # (100, 3, 32, 32) in range [-1, 1] (Tanh)

print(samples.shape)  # torch.Size([100, 3, 32, 32])
```

### Sampling from VAE

```python
from src.models.vae import ConditionalVAE

# Load pretrained VAE
device = torch.device("cuda")
vae = ConditionalVAE()
state = torch.load("outputs/vae/checkpoints/vae_epoch_29.pt")
vae.load_state_dict(state['model_state'])
vae.eval().to(device)

# Sample from latent space
with torch.no_grad():
    y = torch.full((50,), 9).to(device)  # Generate 50 truck images
    samples = vae.sample(50, y, device)
    # samples in range [0, 1] (Sigmoid)
```

## Debugging & Troubleshooting

### Check Logs

```bash
# View real-time logs during training
tail -f outputs/gan/logs/__main__.log

# Or examine after completion
cat outputs/gan/logs/__main__.log
```

### CUDA Memory Issues

**If you get: "RuntimeError: CUDA out of memory"**

```bash
# Modify config
cat > configs/small.yaml << EOF
data:
  batch_size: 8        # Reduced from 32
gan:
  epochs: 20           # Reduced from 50
  generator_hidden_dim: 64    # Reduced from 128
EOF

# Run with reduced settings
python -m src.cli.run_gan --config configs/small.yaml
```

### Slow Training

**If training is very slow:**

1. Use GPU (check: `torch.cuda.is_available()`)
2. Reduce image size in config:
   ```yaml
   data:
     image_size: 16  # from 32
   ```
3. Reduce epochs:
   ```yaml
   gan:
     epochs: 10  # from 50
   ```
4. Check num_workers:
   ```yaml
   data:
     num_workers: 2  # Adjust based on CPU cores
   ```

### Poor Results

**If accuracy is lower than expected:**

1. Increase training iterations:
   ```yaml
   gan:
     epochs: 100
   classifier:
     epochs: 200
   ```

2. Adjust learning rates:
   ```yaml
   gan:
     learning_rate: 0.00005  # Lower lr
   ```

3. Verify data is loading correctly:
   ```python
   from src.data.dataset import get_train_val_datasets
   train_ds, _, _ = get_train_val_datasets()
   img, label = train_ds[0]
   print(img.shape, img.dtype, label)  # Should be torch.float32, value
   ```

## Production Deployment

### Save Trained Models

```bash
# All models already saved in checkpoints/
ls -la outputs/gan/checkpoints/
outputs/gan/checkpoints/
├── gan_epoch_0.pt
├── gan_epoch_1.pt
├── ...
└── classifier_best.pt  # Best classifier checkpoint
```

### Load Production Model

```python
import torch
from src.models.classifier import create_classifier

# Load best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_classifier("resnet18", num_classes=10)

state = torch.load("outputs/gan/checkpoints/classifier_best.pt")
model.load_state_dict(state['model_state'])
model = model.to(device).eval()

# Inference on new images
from torchvision import transforms
import torch.nn.functional as F

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Predict on image
from PIL import Image
img = Image.open("test_image.png")
x = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_class = probs.argmax(1).item()

print(f"Predicted class: {pred_class}, confidence: {probs.max():.2%}")
```

### Batch Inference

```python
from torch.utils.data import DataLoader
from src.data.dataset import get_train_val_datasets

# Load test set
_, _, test_ds = get_train_val_datasets()
test_loader = DataLoader(test_ds, batch_size=128)

# Inference
model.eval()
all_preds = []
all_probs = []

with torch.no_grad():
    for images, _ in test_loader:
        logits = model(images.to(device))
        probs = F.softmax(logits, dim=1)
        all_preds.append(logits.argmax(1).cpu())
        all_probs.append(probs.cpu())

predictions = torch.cat(all_preds)  # shape: (10000,)
probabilities = torch.cat(all_probs)  # shape: (10000, 10)
```

## Performance Optimization

### Enable Mixed Precision (Advanced)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    with autocast():  # Float16 for forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Distributed Training (Multi-GPU)

```python
from torch.nn import DataParallel

model = DataParallel(model)  # Use all available GPUs
```

## Next Steps

1. **Experiment with configurations**: Modify `configs/*.yaml`
2. **Try hybrid approach**: Combine GAN + VAE generations
3. **Implement new methods**: Add diffusion-based oversampling
4. **Extend to other datasets**: CIFAR-100, ImageNet-LT, etc.
5. **Advanced techniques**: Curriculum learning, meta-learning oversampling

---

For more details, see:
- [Dataset Documentation](dataset.md)
- [Methods Documentation](methods.md)
- [Experimental Protocol](experiments.md)
