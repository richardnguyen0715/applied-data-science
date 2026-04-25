# Representation Learning Project

A comprehensive machine learning project for image classification (CIFAR10) and credit card fraud detection.

## Project Structure

```
.
├── configs/                    # Configuration files
│   ├── cifar10_config.yaml
│   └── credit_card_fraud_config.yaml
├── data/                       # Datasets
│   ├── cifar10/               # CIFAR10 dataset
│   └── credit_card_fraud/     # Credit card fraud dataset
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code
│   ├── models/                # Model definitions
│   │   ├── cifar10_models.py
│   │   └── fraud_models.py
│   ├── preprocessing/         # Data preprocessing
│   │   ├── cifar10_processor.py
│   │   └── fraud_processor.py
│   ├── training/              # Training scripts
│   │   └── trainer.py
│   └── utils/                 # Utility functions
│       ├── config.py
│       └── logger.py
├── results/                    # Output results and checkpoints
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Datasets

### CIFAR10
- 60,000 32x32 color images
- 10 classes (airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, trucks)
- Automatically downloaded on first run

### Credit Card Fraud Detection
- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- Binary classification: fraud vs. legitimate
- Download and place `creditcard.csv` in `data/credit_card_fraud/`

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
   - CIFAR10: Automatically downloaded
   - Credit Card Fraud: Download from Kaggle and place in `data/credit_card_fraud/creditcard.csv`

## Configuration

Modify YAML files in `configs/` to adjust:
- Dataset paths and parameters
- Data preprocessing options
- Model architecture
- Training hyperparameters
- Logging settings

## Usage Examples

See `notebooks/` directory for example notebooks or create scripts in `src/training/` following the pattern in `trainer.py`.

## Features

- [x] Modular project structure
- [x] YAML configuration management
- [x] Data preprocessing pipelines
- [x] Model definitions for both datasets
- [x] Logging utilities
- [x] Checkpoint management
- [ ] Training scripts (to implement)
- [ ] Evaluation metrics (to implement)
- [ ] Visualization tools (to implement)

## Requirements

See `requirements.txt` for full list of dependencies.

Key packages:
- PyTorch
- torchvision
- pandas
- scikit-learn
- PyYAML
- numpy

## Notes

- Update `.gitignore` to prevent tracking of data and checkpoints
- Log files are saved in `logs/` directory
- Model checkpoints are saved in `results/*/checkpoints/`
- Ensure you have sufficient disk space for datasets

## License

[Add your license here]
