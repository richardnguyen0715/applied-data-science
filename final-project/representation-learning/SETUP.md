# Project Setup Guide

Quick start guide for setting up the Representation Learning project with CIFAR10 and Credit Card Fraud Detection datasets.

## 🚀 Quick Start

### Option 1: Automated Full Setup (Recommended)

Run this to install everything automatically:

```bash
cd representation-learning
bash scripts/quick_setup.sh
```

This command will:
1. ✓ Install Python dependencies from `requirements.txt`
2. ✓ Download CIFAR10 dataset (~175 MB)
3. ✓ Setup directory structure for results
4. ✓ Verify all datasets
5. ✓ Print helpful next steps

### Option 2: Manual Step-by-Step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and verify datasets
bash scripts/setup_datasets.sh

# 3. Download credit card fraud data (separate)
# See instructions below
```

---

## 📊 Dataset Setup

### CIFAR10 Dataset

**Status**: ✓ Automatic (no action needed)

- **Size**: ~175 MB
- **Samples**: 60,000 (50,000 train + 10,000 test)
- **Auto-downloaded**: Yes
- **Location**: `data/cifar10/`

```bash
# Download automatically
bash scripts/setup_datasets.sh

# Or verify existing download
python3 src/preprocessing/download_datasets.py --verify-only
```

### Credit Card Fraud Detection Dataset

**Status**: ⚠ Manual download required

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and requires authentication to download.

#### Method 1: Using Kaggle CLI (Easiest)

**Step 1: Install Kaggle**
```bash
pip install kaggle
```

**Step 2: Setup Kaggle API credentials**
1. Go to: https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. This downloads `kaggle.json` - save it to:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Step 3: Download dataset**
```bash
bash scripts/download_fraud_dataset.sh
```

Or manually:
```bash
cd data/credit_card_fraud
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
rm creditcardfraud.zip
```

#### Method 2: Manual Download

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" button
3. Extract to: `data/credit_card_fraud/`
4. Verify file is named: `creditcard.csv`

---

## 📁 Project Structure

After setup, your project will look like:

```
representation-learning/
├── scripts/                          # Setup and utility scripts
│   ├── quick_setup.sh               # One-command setup
│   ├── setup_datasets.sh            # Dataset download & verify
│   └── download_fraud_dataset.sh    # Kaggle dataset helper
│
├── src/                              # Source code
│   ├── preprocessing/
│   │   ├── download_datasets.py     # Dataset downloader
│   │   ├── cifar10_processor.py     # CIFAR10 data loader
│   │   └── fraud_processor.py       # Fraud data loader
│   ├── models/
│   ├── training/
│   └── utils/
│
├── configs/                          # Configuration files
│   ├── cifar10_config.yaml
│   └── credit_card_fraud_config.yaml
│
├── data/                             # Datasets
│   ├── cifar10/                     # CIFAR10 (auto-downloaded)
│   └── credit_card_fraud/           # Fraud dataset
│
├── results/                          # Model outputs
│   ├── cifar10/
│   │   ├── checkpoints/
│   │   └── logs/
│   └── credit_card_fraud/
│       ├── checkpoints/
│       └── logs/
│
├── logs/                             # Training logs
├── notebooks/                        # Jupyter notebooks
├── requirements.txt                  # Dependencies
├── SETUP.md                         # This file
└── DATASET_SETUP.md                 # Detailed dataset guide
```

---

## 🔍 Verify Setup

### Check CIFAR10
```bash
python3 src/preprocessing/download_datasets.py --verify-only
```

Expected output:
```
CIFAR10
  Status: READY
  Location: .../data/cifar10
  train_samples: 50000
  test_samples: 10000
  image_size: 32x32
```

### Check Both Datasets
After downloading fraud dataset:
```bash
python3 -c "
from src.preprocessing.cifar10_processor import CIFAR10Processor
from src.preprocessing.fraud_processor import FraudProcessor
from src.utils.config import load_config

# CIFAR10
cfg1 = load_config('configs/cifar10_config.yaml')
proc1 = CIFAR10Processor(cfg1)
train, test = proc1.get_dataloaders(batch_size=32)
print(f'✓ CIFAR10: {len(train)} train batches, {len(test)} test batches')

# Fraud
cfg2 = load_config('configs/credit_card_fraud_config.yaml')
proc2 = FraudProcessor(cfg2)
df = proc2.load_data('data/credit_card_fraud/creditcard.csv')
X, y = proc2.preprocess(df)
print(f'✓ Fraud: {X.shape[0]} samples, {X.shape[1]} features')
"
```

---

## 🔧 Available Scripts

### `quick_setup.sh`
**Complete setup in one command**
```bash
bash scripts/quick_setup.sh
```

Does:
- Install dependencies
- Download CIFAR10
- Setup directories
- Verify everything

### `setup_datasets.sh`
**Download and verify datasets**
```bash
# Full download
bash scripts/setup_datasets.sh

# Skip CIFAR10 if already exists
bash scripts/setup_datasets.sh --skip-cifar10

# Verify only (no download)
bash scripts/setup_datasets.sh --verify-only

# Show help
bash scripts/setup_datasets.sh --help
```

### `download_fraud_dataset.sh`
**Download fraud dataset from Kaggle**
```bash
bash scripts/download_fraud_dataset.sh
```

Requirements:
- Kaggle CLI installed: `pip install kaggle`
- API credentials in: `~/.kaggle/kaggle.json`

### `download_datasets.py`
**Python downloader (used by shell scripts)**
```bash
python3 src/preprocessing/download_datasets.py [OPTIONS]

Options:
  --project-root PATH     Project root directory (default: .)
  --skip-cifar10          Skip CIFAR10 download
  --verify-only           Only verify existing datasets
```

---

## 🐍 Python Usage Examples

### Loading CIFAR10 Data
```python
from src.preprocessing.cifar10_processor import CIFAR10Processor
from src.utils.config import load_config

# Load config
config = load_config('configs/cifar10_config.yaml')

# Initialize processor
processor = CIFAR10Processor(config)

# Get data loaders
train_loader, test_loader = processor.get_dataloaders(batch_size=128)

# Use in training
for images, labels in train_loader:
    print(f"Batch: {images.shape} → {labels.shape}")
    break
```

### Loading Credit Card Fraud Data
```python
from src.preprocessing.fraud_processor import FraudProcessor
from src.utils.config import load_config

# Load config
config = load_config('configs/credit_card_fraud_config.yaml')

# Initialize processor
processor = FraudProcessor(config)

# Load data
df = processor.load_data('data/credit_card_fraud/creditcard.csv')

# Preprocess
X, y = processor.preprocess(df)

# Split
(X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(X, y)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

---

## ⚡ Troubleshooting

### Problem: Dependencies won't install

```bash
# Update pip first
pip install --upgrade pip

# Try again
pip install -r requirements.txt

# Or use specific versions
pip install torch>=2.0.0 torchvision>=0.15.0 pandas scikit-learn pyyaml
```

### Problem: CIFAR10 download fails

```bash
# Check internet connection, then retry
bash scripts/setup_datasets.sh --skip-cifar10

# Or download manually
python3 -c "from torchvision import datasets; datasets.CIFAR10('.', download=True)"
```

### Problem: Kaggle API not working

```bash
# Verify credentials are set up
ls -la ~/.kaggle/kaggle.json

# Check Kaggle CLI works
kaggle datasets list | head

# Try downloading manually from Kaggle website instead
```

### Problem: "File not found" errors

```bash
# Check data directory structure
ls -la data/cifar10/
ls -la data/credit_card_fraud/

# Verify file names (case-sensitive)
ls data/credit_card_fraud/creditcard.csv

# Re-download if corrupted
rm -rf data/credit_card_fraud/
bash scripts/download_fraud_dataset.sh
```

### Problem: Out of disk space

CIFAR10 needs ~175 MB, fraud dataset ~60 MB. Ensure:
```bash
# Check free space
df -h /

# Free up space if needed
rm -rf ~/.cache/pip  # Clear pip cache
```

---

## 📚 Next Steps

After successful setup:

1. **Explore notebooks**: Check `notebooks/` for examples
2. **Review configs**: Edit `configs/*.yaml` for your needs
3. **Start training**: Implement training scripts in `src/training/`
4. **Save results**: Outputs go to `results/` directory

---

## 📋 System Requirements

- **Python**: 3.8+
- **RAM**: 8 GB minimum (16 GB recommended for training)
- **Disk**: 500 MB (200 MB datasets + 300 MB dependencies)
- **Internet**: Required for first-time downloads

---

## 📝 Configuration Files

### CIFAR10 Config (`configs/cifar10_config.yaml`)

```yaml
dataset:
  name: cifar10
  num_classes: 10
  image_size: 32

preprocessing:
  normalize:
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2023, 0.1994, 0.2010]
  augmentation:
    enabled: true

training:
  batch_size: 128
  num_epochs: 100
  learning_rate: 0.1
```

### Fraud Config (`configs/credit_card_fraud_config.yaml`)

```yaml
dataset:
  name: credit_card_fraud
  csv_path: data/credit_card_fraud/creditcard.csv

preprocessing:
  test_size: 0.2
  scaling: standardscaler
  handle_imbalance: true

training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.001
```

Modify these files to customize your setup!

---

## 🆘 Getting Help

1. **Read**: `DATASET_SETUP.md` for detailed dataset info
2. **Check**: Logs in `logs/` directory
3. **Verify**: Dataset integrity with `--verify-only` flag
4. **Debug**: Run individual components to isolate issues

---

## ✅ Verification Checklist

After running `quick_setup.sh`, verify:

- [ ] Dependencies installed successfully
- [ ] CIFAR10 dataset downloaded (~175 MB in `data/cifar10/`)
- [ ] `data/credit_card_fraud/` exists and is empty (awaiting manual download)
- [ ] Result directories created in `results/`
- [ ] Log directories created in `logs/`
- [ ] All config files present in `configs/`

Once you download the Kaggle dataset:
- [ ] `creditcard.csv` placed in `data/credit_card_fraud/`
- [ ] File size ~60 MB
- [ ] Verification passes: `bash scripts/setup_datasets.sh --verify-only`

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review detailed logs: `python3 src/preprocessing/download_datasets.py --help`
3. See DATASET_SETUP.md for dataset-specific help

Good luck! 🚀

