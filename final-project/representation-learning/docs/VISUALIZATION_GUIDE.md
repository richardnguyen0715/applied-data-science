# Dataset Visualization Pipeline Guide

## Overview

The **Dataset Visualization Pipeline** provides automated statistical analysis and visualization for both CIFAR10 and Credit Card Fraud Detection datasets. It generates publication-quality visualizations and comprehensive statistical reports.

## 🚀 Quick Start

```bash
# One-command visualization
bash scripts/visualize_datasets.sh

# Or run directly with Python
python3 src/visualization/dataset_analyzer.py
```

## 📊 What Gets Generated

### Visualizations (7 PNG files @ 300 DPI)

#### CIFAR10 Dataset
1. **cifar10_class_distribution.png**
   - Training vs Test set class distribution
   - Bar charts showing sample count per class
   - All 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

2. **cifar10_sample_images.png**
   - 12 random sample images from training set
   - Labeled with class names
   - Shows actual image quality and characteristics

3. **cifar10_channel_statistics.png**
   - RGB channel mean values
   - RGB channel standard deviations
   - Useful for normalization parameters

#### Credit Card Fraud Dataset
4. **fraud_class_distribution.png**
   - Legitimate vs Fraudulent transactions
   - Absolute counts and percentages
   - Highlights class imbalance

5. **fraud_amount_distribution.png**
   - Overall transaction amount distribution
   - Separate histograms for legitimate vs fraud
   - Shows amount patterns

6. **fraud_correlation_heatmap.png**
   - Feature correlation matrix (30x30)
   - Color-coded correlation strength
   - Identifies feature relationships

7. **fraud_time_patterns.png**
   - Transaction volume over time (in hours)
   - Fraud rate temporal patterns
   - Shows temporal dynamics

### Statistics (JSON Files)

#### cifar10_statistics.json
```json
{
  "dataset": "CIFAR10",
  "train_samples": 50000,
  "test_samples": 10000,
  "total_samples": 60000,
  "image_shape": [3, 32, 32],
  "num_classes": 10,
  "classes": ["airplane", "automobile", ...],
  "class_distribution_train": [5000, 5000, ...],
  "class_distribution_test": [1000, 1000, ...],
  "channel_means": [0.4914, 0.4822, 0.4465],
  "channel_stds": [0.2023, 0.1994, 0.2010],
  "channel_names": ["Red", "Green", "Blue"]
}
```

#### fraud_statistics.json
```json
{
  "dataset": "Credit Card Fraud Detection",
  "total_transactions": 284807,
  "total_features": 31,
  "memory_usage_mb": 67.89,
  "class_distribution": {
    "legitimate": 284315,
    "fraud": 492,
    "fraud_percentage": 0.173
  },
  "feature_statistics": {
    "amount_mean": 88.35,
    "amount_median": 22.0,
    "amount_std": 250.12,
    "amount_min": 0.0,
    "amount_max": 25691.16,
    "time_range_seconds": 172792
  },
  "missing_values": {...}
}
```

### Summary Report

**VISUALIZATION_REPORT.md** - Comprehensive markdown report including:
- Basic statistics for both datasets
- Channel statistics (CIFAR10)
- Class distribution details (Fraud)
- Amount statistics (Fraud)
- File listings and locations
- Key insights

## 📁 File Structure

```
project/
├── scripts/
│   ├── visualize_datasets.sh       # Main visualization script (execute this)
│   └── visualize_quick.sh          # Quick analysis (optional)
│
├── src/visualization/
│   ├── __init__.py                 # Module initialization
│   └── dataset_analyzer.py         # Core analyzer class (20KB)
│
└── results/
    ├── visualizations/             # PNG files (300 DPI)
    │   ├── cifar10_class_distribution.png
    │   ├── cifar10_sample_images.png
    │   ├── cifar10_channel_statistics.png
    │   ├── fraud_class_distribution.png
    │   ├── fraud_amount_distribution.png
    │   ├── fraud_correlation_heatmap.png
    │   └── fraud_time_patterns.png
    │
    ├── statistics/                 # JSON files
    │   ├── cifar10_statistics.json
    │   └── fraud_statistics.json
    │
    └── VISUALIZATION_REPORT.md     # Summary report
```

## 🔧 Usage

### Using the Bash Script (Recommended)

```bash
# Navigate to project root
cd /path/to/project

# Run visualization pipeline
bash scripts/visualize_datasets.sh
```

The script will:
1. ✅ Check Python 3 installation
2. ✅ Verify required packages (matplotlib, seaborn, pandas, numpy)
3. ✅ Check dataset availability
4. ✅ Create output directories
5. ✅ Run the analyzer
6. ✅ Generate visualizations and statistics
7. ✅ Display summary

### Using Python Directly

```python
from src.visualization.dataset_analyzer import DatasetAnalyzer

# Initialize analyzer
analyzer = DatasetAnalyzer(
    data_dir="./data",
    results_dir="./results"
)

# Analyze CIFAR10
cifar10_stats = analyzer.analyze_cifar10()
print(f"CIFAR10: {cifar10_stats['total_samples']} samples")

# Analyze Fraud
fraud_stats = analyzer.analyze_fraud()
print(f"Fraud: {fraud_stats['total_transactions']} transactions")

# Generate report
analyzer.generate_summary_report(cifar10_stats, fraud_stats)
```

### Using Python CLI

```bash
python3 src/visualization/dataset_analyzer.py
```

## 📋 Customization

Edit `src/visualization/dataset_analyzer.py` to customize:

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| Number of sample images | Line ~227 | 12 (3x4) | Change `num_samples` parameter |
| Histogram bins | Line ~363 | 100 | Adjust granularity |
| Correlation heatmap size | Line ~401 | 16x12 | Change `figsize` parameter |
| Time period bins | Line ~418 | 50 | Change `time_bins` granularity |
| DPI (image quality) | Line ~235 | 300 | Increase for higher quality |

## 🐛 Troubleshooting

### Missing CIFAR10 Dataset
```
⚠️  WARNING: CIFAR10 dataset not found at ./data/cifar-10-batches-py
```
**Solution:** The analyzer will attempt to auto-download during analysis. First time may take 5-10 minutes.

### Missing Fraud Dataset
```
⚠️  WARNING: Fraud dataset not found at ./data/creditcardfraud/creditcard.csv
```
**Solution:** Download using:
```bash
bash scripts/download_fraud_dataset.sh
```

### Import Errors
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Permission Denied
```
bash: scripts/visualize_datasets.sh: Permission denied
```
**Solution:** Make script executable:
```bash
chmod +x scripts/visualize_datasets.sh
```

### Out of Memory
For large fraud dataset (280K+ rows):
- Use a machine with 4GB+ RAM
- Or analyze subsets of the data
- The script efficiently samples data for statistics

## 📊 Outputs Explained

### Class Distribution Charts
- **Purpose:** Identify class imbalance
- **CIFAR10:** Shows balanced distribution (5000 samples/class)
- **Fraud:** Shows severe imbalance (0.17% fraud rate)

### Sample Images (CIFAR10 Only)
- **Purpose:** Visual quality assessment
- **Shows:** Actual image data, not synthetic
- **Size:** 32×32 RGB, typical ML training data

### Channel Statistics (CIFAR10 Only)
- **Purpose:** Normalization parameters
- **Use in preprocessing:** 
  ```python
  transform = transforms.Normalize(
      mean=[0.4914, 0.4822, 0.4465],
      std=[0.2023, 0.1994, 0.2010]
  )
  ```

### Correlation Heatmap (Fraud Only)
- **Purpose:** Feature relationship analysis
- **Red:** Positive correlation
- **Blue:** Negative correlation
- **White:** No correlation

### Time Patterns (Fraud Only)
- **Purpose:** Temporal trend detection
- **Shows:** When fraud occurs
- **Use case:** Time-based fraud detection features

## ⏱️ Performance

| Dataset | Processing Time | Output Size | Memory Used |
|---------|-----------------|-------------|-------------|
| CIFAR10 | 30-60 seconds | 2-3 MB (PNGs) | 500 MB |
| Fraud | 20-30 seconds | 8-12 MB (PNGs) | 200 MB |
| **Total** | **1-2 minutes** | **10-15 MB** | **700 MB** |

*Times vary based on system performance*

## 🎯 Use Cases

1. **Data Exploration**
   - Understand dataset characteristics before modeling
   - Identify imbalance and distribution issues

2. **Report Generation**
   - Create publication-quality figures
   - Document data characteristics

3. **Feature Engineering**
   - Use correlation heatmaps to select features
   - Identify redundant features

4. **Baseline Comparison**
   - Compare datasets across projects
   - Track changes in data preprocessing

5. **Presentations**
   - Use visualizations in slides/reports
   - 300 DPI PNG suitable for publications

## 📚 Related Guides

- **SETUP.md** - Initial project setup
- **DATASET_SETUP.md** - Downloading and verifying datasets
- **QUICK_REFERENCE.txt** - Command cheat sheet
- **PIPELINE_SUMMARY.md** - Overall project architecture

## 🔗 Dependencies

Required packages (all in `requirements.txt`):
- `torch` / `torchvision` - CIFAR10 dataset loading
- `pandas` - Fraud dataset handling
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `seaborn` - Statistical graphics
- `scikit-learn` - Statistical functions

## 📝 Output Example

```
======================================================================
🔍 DATASET VISUALIZATION PIPELINE
======================================================================

ℹ️  INFO: Performing pre-flight checks...
✅ SUCCESS: Python 3 found: Python 3.9.13
✅ SUCCESS: All required packages are installed
✅ SUCCESS: Output directories ready

======================================================================
🚀 RUNNING VISUALIZATION ANALYSIS
======================================================================

ℹ️  INFO: Starting CIFAR10 analysis...
✅ CIFAR10 analysis complete!
ℹ️  INFO: Starting Credit Card Fraud analysis...
✅ Credit Card Fraud analysis complete!

======================================================================
📊 VISUALIZATION COMPLETE
======================================================================

📁 Visualizations:
   - results/visualizations/cifar10_class_distribution.png
   - results/visualizations/cifar10_sample_images.png
   - results/visualizations/cifar10_channel_statistics.png
   - results/visualizations/fraud_class_distribution.png
   - results/visualizations/fraud_amount_distribution.png
   - results/visualizations/fraud_correlation_heatmap.png
   - results/visualizations/fraud_time_patterns.png

📊 Statistics (JSON):
   - results/statistics/cifar10_statistics.json
   - results/statistics/fraud_statistics.json

📄 Report:
   - results/VISUALIZATION_REPORT.md
```

## ✅ Verification

After running the pipeline, verify outputs:

```bash
# Check visualizations exist
ls -lh results/visualizations/
# Output: 7 PNG files, 2-3 MB each

# Check statistics exist
ls -lh results/statistics/
# Output: 2 JSON files

# Check report
cat results/VISUALIZATION_REPORT.md

# Verify image quality (macOS)
open results/visualizations/
```

## 📞 Support

For issues:
1. Check **Troubleshooting** section above
2. Verify datasets are downloaded: `bash scripts/download_fraud_dataset.sh`
3. Verify dependencies: `pip install -r requirements.txt`
4. Check logs: `cat logs/analyzer.log`

---

**Version:** 1.0  
**Last Updated:** 2026-04-25  
**Status:** ✅ Production Ready
