# 📊 Dataset Visualization Pipeline

## Overview

Complete automated statistical visualization and analysis system for **CIFAR10** and **Credit Card Fraud Detection** datasets. Generate publication-quality visualizations and comprehensive statistics with a single command.

## ✨ Features

✅ **Automated Analysis**
- CIFAR10: Class distribution, sample images, channel statistics
- Fraud Detection: Class imbalance, amounts, correlations, time patterns

✅ **High-Quality Outputs**
- 7 publication-ready PNG files (300 DPI)
- 2 JSON statistics files
- Comprehensive markdown report

✅ **Easy to Use**
- One-command execution
- Automatic dataset downloading (CIFAR10)
- Minimal dependencies
- Comprehensive error handling

✅ **Well-Documented**
- Complete user guide
- Bash script with color output
- Inline code documentation
- Troubleshooting guide

## 🚀 Quick Start

```bash
# Run visualization pipeline
bash scripts/visualize_datasets.sh

# Or quick analysis (faster, less verbose)
bash scripts/visualize_quick.sh

# Or run directly with Python
python3 src/visualization/dataset_analyzer.py
```

## 📁 Created Files

### Core Modules (Python)
- **src/visualization/dataset_analyzer.py** (21 KB)
  - Main analyzer class
  - CIFAR10 and Fraud analysis methods
  - Visualization functions
  - Statistics generation

- **src/visualization/config.py** (3 KB)
  - Configuration management
  - Customizable parameters
  - Default settings

- **src/visualization/__init__.py**
  - Module initialization

### Bash Scripts
- **scripts/visualize_datasets.sh** (5 KB)
  - Main entry point
  - Pre-flight checks
  - Error handling
  - Detailed logging

- **scripts/visualize_quick.sh** (2 KB)
  - Faster variant
  - Less verbose output
  - Same analysis

- **scripts/test_visualization.sh** (6 KB)
  - Validation test suite
  - 10 comprehensive tests
  - System diagnostics

### Documentation
- **VISUALIZATION_GUIDE.md** (11 KB)
  - Complete user guide
  - Usage examples
  - Customization options
  - Troubleshooting

## 📊 Output Files

After running the pipeline, you'll get:

### Visualizations (PNG, 300 DPI)
```
results/visualizations/
├── cifar10_class_distribution.png      # Train/test class distribution
├── cifar10_sample_images.png           # 12 random sample images
├── cifar10_channel_statistics.png      # RGB channel statistics
├── fraud_class_distribution.png        # Legitimate vs fraud split
├── fraud_amount_distribution.png       # Transaction amounts
├── fraud_correlation_heatmap.png       # 30x30 feature correlations
└── fraud_time_patterns.png             # Temporal patterns
```

### Statistics (JSON)
```
results/statistics/
├── cifar10_statistics.json             # CIFAR10 metrics (50K/10K samples)
└── fraud_statistics.json               # Fraud metrics (284K transactions)
```

### Report
```
results/
└── VISUALIZATION_REPORT.md             # Comprehensive analysis report
```

## 📋 What Gets Analyzed

### CIFAR10 Dataset
| Metric | Value |
|--------|-------|
| Training Samples | 50,000 |
| Test Samples | 10,000 |
| Image Shape | 32×32 RGB |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Class Balance | Perfectly balanced (5,000 per class) |
| Generated Visualizations | 3 PNG files |

**Outputs:**
- Channel means and standard deviations (for normalization)
- Per-class sample counts
- Sample image gallery

### Credit Card Fraud Dataset
| Metric | Value |
|--------|-------|
| Total Transactions | ~284,000 |
| Features | 31 |
| Legitimate | ~284,300 (99.83%) |
| Fraudulent | ~500 (0.17%) |
| Imbalance Ratio | ~579:1 |
| Generated Visualizations | 4 PNG files |

**Outputs:**
- Class distribution (highly imbalanced)
- Transaction amount statistics
- Feature correlations (heatmap)
- Time-based patterns

## 🛠️ Usage Examples

### Basic Usage
```bash
cd /path/to/project
bash scripts/visualize_datasets.sh
```

### Python API
```python
from src.visualization.dataset_analyzer import DatasetAnalyzer

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

### Custom Configuration
```python
from src.visualization.config import VisualizationConfig

config = VisualizationConfig()

# Get settings
num_images = config.get("cifar10", "num_sample_images")  # 12
dpi = config.get("cifar10", "dpi")  # 300

# Modify settings
config.config["cifar10"]["num_sample_images"] = 20
config.save_config("custom_config.json")
```

## ✅ Pre-flight Checks

The bash script automatically verifies:
- ✅ Python 3 installation
- ✅ Required packages (matplotlib, seaborn, pandas, numpy)
- ✅ Dataset availability
- ✅ Output directories
- ✅ Script permissions

## ⏱️ Performance

| Component | Time | Memory |
|-----------|------|--------|
| CIFAR10 Analysis | 30-60 sec | 500 MB |
| Fraud Analysis | 20-30 sec | 200 MB |
| Total Pipeline | 1-2 min | 700 MB |

## 🐛 Troubleshooting

### Missing CIFAR10
```
⚠️  CIFAR10 not found - will auto-download on first run
```
**Solution:** The analyzer will automatically download on first execution (takes 5-10 minutes for initial download)

### Missing Fraud Dataset
```
⚠️  Fraud dataset not found
```
**Solution:** Download using:
```bash
bash scripts/download_fraud_dataset.sh
```

### Import Error
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Permission Denied
```
bash: visualize_datasets.sh: Permission denied
```
**Solution:** Make scripts executable:
```bash
chmod +x scripts/*.sh
```

## 📚 File Documentation

| File | Size | Purpose |
|------|------|---------|
| dataset_analyzer.py | 21 KB | Core analyzer engine |
| config.py | 3 KB | Configuration management |
| visualize_datasets.sh | 5 KB | Main entry point |
| visualize_quick.sh | 2 KB | Fast analysis |
| test_visualization.sh | 6 KB | Test suite |
| VISUALIZATION_GUIDE.md | 11 KB | User guide |

## 🔧 Customization

### Change Number of Sample Images
Edit `src/visualization/dataset_analyzer.py`, line ~227:
```python
def _plot_cifar10_samples(self, dataset, num_samples: int = 20):  # Change from 12
```

### Change Histogram Bins
Edit `src/visualization/dataset_analyzer.py`, line ~363:
```python
axes[0].hist(df['Amount'], bins=200, ...)  # Change from 100
```

### Change DPI (Quality)
Edit `src/visualization/config.py`:
```python
"dpi": 600,  # Higher = better quality = larger file
```

## 🎯 Use Cases

1. **Data Exploration**
   - Understand dataset characteristics
   - Identify distribution issues
   - Spot imbalance problems

2. **Report Generation**
   - Create publication-quality figures
   - Document data preprocessing
   - Share with team

3. **Feature Engineering**
   - Use correlations to select features
   - Identify redundant features
   - Detect multicollinearity

4. **Model Preparation**
   - Understand baseline statistics
   - Set normalization parameters
   - Detect class imbalance issues

5. **Presentations**
   - Professional visualizations
   - 300 DPI suitable for print
   - Markdown report for documentation

## 📞 Support

For issues:
1. Check **VISUALIZATION_GUIDE.md** for detailed documentation
2. Run test suite: `bash scripts/test_visualization.sh`
3. Check logs: `cat logs/DatasetAnalyzer.log`
4. Verify datasets: `ls -la data/`

## 🎓 Learning Resources

- **CIFAR10:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Fraud Detection:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Matplotlib:** https://matplotlib.org/
- **Seaborn:** https://seaborn.pydata.org/

## 📝 License

This visualization pipeline is part of the Applied Data Science project.

## ✨ Next Steps

After visualization:
1. ✅ Explore the generated visualizations
2. ✅ Read the statistics JSON files
3. ✅ Review the markdown report
4. ✅ Use insights for model training
5. ✅ Share visualizations in presentations

---

**Status:** ✅ Production Ready  
**Last Updated:** 2026-04-25  
**Version:** 1.0
