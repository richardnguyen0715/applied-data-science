# Dataset Setup Guide

This guide explains how to download and verify the datasets for the Representation Learning project.

## Datasets Overview

### CIFAR10
- **Source**: Automatically downloaded from torchvision
- **Size**: ~175 MB
- **Samples**: 60,000 (50,000 train + 10,000 test)
- **Auto-download**: Yes ✓

### Credit Card Fraud Detection
- **Source**: Kaggle
- **Dataset**: [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~60 MB
- **Samples**: 284,807 transactions
- **Manual Setup**: Required (see below)

---

## Quick Start (Automatic)

### Option 1: Full Setup (Recommended)
Run the quick setup script to install dependencies and setup all datasets:

```bash
cd representation-learning
bash scripts/quick_setup.sh
```

This will:
1. ✓ Install Python dependencies
2. ✓ Download and verify CIFAR10
3. ✓ Verify existing datasets or provide instructions

### Option 2: Dataset-Only Setup
If dependencies are already installed:

```bash
bash scripts/setup_datasets.sh
```

### Option 3: Verify Only
To verify existing datasets without downloading:

```bash
bash scripts/setup_datasets.sh --verify-only
```

---

## Manual Setup for Credit Card Fraud Dataset

The Credit Card Fraud dataset requires manual download due to Kaggle's terms of service.

### Method 1: Kaggle CLI (Recommended)

**Prerequisites:**
```bash
pip install kaggle
```

**Setup Credentials:**
1. Go to: https://www.kaggle.com/settings/account
2. Click "Create New API Token" (downloads `kaggle.json`)
3. Move the file:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Download Dataset:**
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

### Method 2: Manual Download

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" button
3. Extract the ZIP file to `data/credit_card_fraud/`
4. Verify the file structure (see below)

---

## Verify Dataset Setup

After downloading, verify all datasets:

```bash
python3 src/preprocessing/download_datasets.py --verify-only
```

Or via the shell script:
```bash
bash scripts/setup_datasets.sh --verify-only
```

Expected output:
```
CIFAR10
  Status: READY
  Location: /path/to/data/cifar10
  train_samples: 50000
  test_samples: 10000
  image_size: 32x32
  channels: 3
  classes: 10

CREDIT CARD FRAUD
  Status: READY
  Location: /path/to/data/credit_card_fraud
  transactions: 284807
  features: 31
  fraud_cases: 492
  fraud_rate: 0.17%
```

---

## Directory Structure

After successful setup:

```
data/
├── cifar10/
│   ├── cifar-10-batches-py/
│   │   ├── batches.meta
│   │   ├── data_batch_1-5
│   │   └── test_batch
│   └── cifar-10-python.tar.gz (optional)
│
└── credit_card_fraud/
    ├── creditcard.csv              # Main dataset file
    └── README.md                   # Dataset documentation (if available)
```

---

## Python API Usage

### Loading CIFAR10

```python
from src.preprocessing.cifar10_processor import CIFAR10Processor
from src.utils.config import load_config

# Load configuration
config = load_config('configs/cifar10_config.yaml')

# Initialize processor
processor = CIFAR10Processor(config)

# Get dataloaders
train_loader, test_loader = processor.get_dataloaders(batch_size=128)

# Iterate
for images, labels in train_loader:
    print(images.shape)  # torch.Size([128, 3, 32, 32])
    print(labels.shape)  # torch.Size([128])
    break
```

### Loading Credit Card Fraud Data

```python
from src.preprocessing.fraud_processor import FraudProcessor
from src.utils.config import load_config

# Load configuration
config = load_config('configs/credit_card_fraud_config.yaml')

# Initialize processor
processor = FraudProcessor(config)

# Load and preprocess
df = processor.load_data('data/credit_card_fraud/creditcard.csv')
X_scaled, y = processor.preprocess(df)

# Split data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_data(X_scaled, y)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
# Train: (199765, 30), Val: (42521, 30), Test: (42521, 30)
```

---

## Troubleshooting

### CIFAR10 Download Issues

**Problem**: "Connection timeout"
- **Solution**: 
  - Check internet connection
  - Try running again (downloads resume)
  - Manual download: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

**Problem**: "Disk space insufficient"
- **Solution**: Need ~200 MB free space after download

### Credit Card Fraud Dataset Issues

**Problem**: "File not found after extraction"
- **Solution**: 
  - Re-download from Kaggle
  - Check file is named exactly `creditcard.csv` (case-sensitive)

**Problem**: "Kaggle API not configured"
- **Solution**:
  - Follow Method 1 setup instructions above
  - Ensure `~/.kaggle/kaggle.json` exists with correct permissions

**Problem**: "Verification fails - corrupted data"
- **Solution**:
  - Delete dataset files
  - Re-download fresh copy
  - Check file integrity with MD5 hash (provided on Kaggle)

---

## Script Options

### `setup_datasets.sh`

```bash
./scripts/setup_datasets.sh [OPTIONS]

Options:
  --skip-cifar10      Skip CIFAR10 download if already exists
  --verify-only       Only verify existing datasets without downloading
  --help              Show help message
```

### `quick_setup.sh`

```bash
./scripts/quick_setup.sh
```

Runs complete setup: dependencies → datasets → verification

### `download_fraud_dataset.sh`

```bash
./scripts/download_fraud_dataset.sh
```

Downloads and extracts Credit Card Fraud dataset using Kaggle CLI

---

## Advanced: Custom Download Script

Create a custom Python script:

```python
from src.preprocessing.download_datasets import DatasetDownloader

# Initialize
downloader = DatasetDownloader(project_root='.')

# Download everything
success, summary = downloader.run()

# Or individual operations
downloader.download_cifar10()
downloader.verify_cifar10()

# Check status
summary = downloader.generate_summary()
print(summary)
```

---

## Dataset Statistics

### CIFAR10
| Metric | Value |
|--------|-------|
| Training Samples | 50,000 |
| Test Samples | 10,000 |
| Image Size | 32×32 pixels |
| Channels | 3 (RGB) |
| Classes | 10 |
| Class Distribution | Balanced |

### Credit Card Fraud
| Metric | Value |
|--------|-------|
| Total Transactions | 284,807 |
| Features | 30 (V1-V28 + Time + Amount) |
| Fraudulent Cases | 492 |
| Legitimate Cases | 284,315 |
| Fraud Rate | 0.17% |
| Class Imbalance Ratio | 1:579 |

---

## Notes

- **CIFAR10**: Downloaded automatically on first run (~175 MB)
- **Credit Card Fraud**: Manual download required (~60 MB)
- Datasets are cached after first download
- Verification checks data integrity and structure
- All download/verify operations log detailed information

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the detailed logs in `logs/`
3. Run: `python3 src/preprocessing/download_datasets.py --help`
4. Verify your internet connection and disk space

