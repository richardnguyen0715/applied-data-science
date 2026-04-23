# Data Directory

This directory is for storing datasets used in the contrastive learning pipeline.

## Credit Card Fraud Detection Dataset

### Download Instructions

1. Visit Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Download `creditcard.csv` (approximately 152 MB)

3. Place the file in this directory:
   ```
   data/
   └── creditcard.csv
   ```

### Dataset Information

- **Size**: 284,807 transactions
- **Features**: 28 principal component analysis (PCA) features + Amount
- **Target**: Class (0 = Normal, 1 = Fraud)
- **Imbalance**: Highly imbalanced (~0.17% fraud cases)

### Usage

Once the file is in place, you can train with:

```bash
python main.py --dataset credit-card-fraud
```

Or:

```bash
bash src/cli/train_creditcard.sh
```

## CIFAR-10-LT Dataset

The CIFAR-10-LT dataset is automatically downloaded from HuggingFace Datasets when you run training.
No manual download is required.

### Available Configurations

- `r-10`: Imbalance ratio 10:1
- `r-20`: Imbalance ratio 20:1
- `r-50`: Imbalance ratio 50:1
- `r-100`: Imbalance ratio 100:1 (default)

### Usage

```bash
python main.py --dataset cifar10-lt --cifar10-config r-100
```

## Notes

- Large datasets may take time to download on first use
- Ensure you have sufficient disk space (at least 500 MB)
- For Credit Card Fraud dataset, you need a Kaggle account to download
