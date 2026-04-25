# Comparison Pipeline - Component Index

## 📋 Overview

Complete pipeline untuk analyze dan visualize imbalance handling effectiveness:
- **Training**: M2M synthesis augmentation untuk minority classes
- **Evaluation**: Comprehensive before/after comparison
- **Visualization**: Training curves, per-class analysis, confusion matrix
- **Export**: CSV metrics, PNG plots, detailed report

---

## 🔗 Components

### 1. **Jupyter Notebook** (Interactive Analysis)
📍 [`notebooks/comparison_pipeline.ipynb`](../notebooks/comparison_pipeline.ipynb)

**11 analysis sections:**
- Problem intro & setup
- Load config/data
- Training history
- Model evaluation
- Training curves visualization
- Per-class performance
- Confusion matrix
- Imbalance impact analysis
- M2M contribution assessment
- Summary report
- CSV/PNG export

**Run:**
```bash
jupyter notebook notebooks/comparison_pipeline.ipynb
```

**Output:** Inline plots + 3 PNG files + 3 CSV files

---

### 2. **Python Utility Module** (Reusable Functions)
📍 [`src/utils/comparison.py`](../src/utils/comparison.py)

**Classes:**
- `ImbalanceComparisonAnalyzer`: Load history, checkpoint, evaluate, generate reports
- Functions: `create_comparison_plots()` - Generate all 3 visualizations

**Usage:**
```python
from src.utils.comparison import ImbalanceComparisonAnalyzer

analyzer = ImbalanceComparisonAnalyzer(config, device, results_dir)
history_df, success = analyzer.load_training_history()
model, checkpoint, success = analyzer.load_checkpoint()
results = analyzer.evaluate_and_compare(model, test_loader, train_dist)
report = analyzer.generate_summary_report(history_df, results, train_dist)
```

**Methods:**
- `load_training_history()` - Returns (DataFrame, bool)
- `load_checkpoint(checkpoint_name='best.pt')` - Returns (model, checkpoint_dict, bool)
- `evaluate_and_compare(model, test_loader, train_dist)` - Returns metrics dict
- `generate_summary_report(history_df, results, train_dist)` - Returns report string
- `save_analysis_results(results, output_dir)` - Save CSVs

---

### 3. **Python Script** (Command Line)
📍 [`src/scripts/run_comparison.py`](../src/scripts/run_comparison.py)

**Standalone script to run full pipeline:**

```bash
# Default paths
python src/scripts/run_comparison.py

# Custom paths
python src/scripts/run_comparison.py \
  --config src/configs/config.yaml \
  --results-dir results/cifar10 \
  --output-dir results/comparison
```

**Features:**
- Argparse CLI with custom path support
- Logging to file
- Loads history, checkpoint, evaluates model
- Generates report and plots
- Saves CSVs to output directory

---

### 4. **Shell Script Wrapper** (Easy Runner)
📍 [`src/scripts/run_comparison.sh`](../src/scripts/run_comparison.sh)

**Three modes:**

```bash
# Interactive notebook (default)
bash src/scripts/run_comparison.sh notebook
# or
bash src/scripts/run_comparison.sh

# Script mode (no UI)
bash src/scripts/run_comparison.sh script

# View latest results
bash src/scripts/run_comparison.sh view-results

# Help
bash src/scripts/run_comparison.sh help
```

**Features:**
- Virtual env auto-activation
- Color output
- Checks for Jupyter, installs if needed
- Auto-paths to results directory

---

### 5. **Documentation**

#### **Quick Start Guide** 👈 **START HERE**
📍 [`docs/COMPARISON_QUICKSTART.md`](../docs/COMPARISON_QUICKSTART.md)

**Contains:**
- Purpose & goals
- Prerequisites
- 3 ways to run (notebook/script/CLI)
- Output file descriptions
- How to read results
- Detailed analysis steps
- Troubleshooting
- Quick commands reference

#### **Full Documentation**
📍 [`docs/COMPARISON_PIPELINE.md`](../docs/COMPARISON_PIPELINE.md)

**Contains:**
- Complete overview
- Detailed metrics explanation (Accuracy, Balanced Accuracy, Per-Class)
- Output structure
- How to interpret plots
- Expected results benchmark
- Customization guide
- FAQ

---

## 📊 Output Files Generated

After running pipeline, generates:

```
results/comparison/
├── logs/
│   └── comparison.log              # Detailed report
├── per_class_performance.csv        # Accuracy + train counts per class
├── overall_metrics.csv              # Summary stats
├── training_curves.png              # 2x2: loss, acc, balanced_acc, M2M activity
├── per_class_accuracy.png           # Per-class bars with ref lines
└── imbalance_vs_accuracy.png        # Imbalance distribution vs accuracy
```

---

## 🔄 Workflow

### Step 1: Run Training (if not done)
```bash
bash src/scripts/run_train.sh
# Creates: results/cifar10/logs/history.json
#          results/cifar10/checkpoints/best.pt
```

### Step 2: Run Comparison
Choose one:

```bash
# Option A: Interactive exploration
bash src/scripts/run_comparison.sh notebook

# Option B: Batch processing
bash src/scripts/run_comparison.sh script

# Option C: Direct Python
python src/scripts/run_comparison.py
```

### Step 3: View & Analyze Results
```bash
# View report
cat results/comparison/logs/comparison.log

# View plots (open in image viewer)
open results/comparison/*.png

# Analyze CSVs
head results/comparison/*.csv
```

---

## 🎯 Key Metrics Explained

| Metric | Why It Matters | Good Value |
|--------|---------------|-----------  |
| **Standard Accuracy** | Overall correctness | High, but can be misleading for imbalanced data |
| **Balanced Accuracy** | Fair per-class average | >50% improvement from baseline |
| **Per-Class Accuracy** | Individual class performance | Minority classes >30% (vs 10% random) |
| **Std Dev** | Performance variance | Lower = more balanced across classes |

### Example Interpretation

```
Minority class (50 samples):  accuracy = 0.41 (vs 0.05 baseline)  ← M2M works!
Majority class (5000 samples): accuracy = 0.82 (vs 0.80 baseline) ← Minimal change

Balanced Accuracy: 0.58 (vs 0.35 baseline)                        ← +23% gain!
```

---

## 🛠️ Customization

### Change Imbalance Factor
Edit `src/configs/config.yaml`:
```yaml
imbalance:
  imbalance_factor: 0.01  # 1:100 ratio
  # Change to 0.05 for 1:20, 0.1 for 1:10, etc
```

### Change Output Directory
```bash
python src/scripts/run_comparison.py \
  --output-dir custom_results/
```

### Add Custom Analysis
Modify notebook cells or extend `comparison.py` module.

---

## 📈 Expected Results

After M2M training on CIFAR-10 (imbalance_factor=0.01):

```
Dataset:
  Majority class:    5000 samples
  Minority class:    50 samples

Standard Accuracy:     0.68 ↑ from 0.65
Balanced Accuracy:     0.58 ↑ from 0.35 (+23% improvement)
Per-class Std Dev:     0.08 ↓ from 0.20 (more balanced)
Worst class accuracy:  0.41 ↑ from 0.05 (+36% improvement)

M2M Synthesis:
  Active epochs:    55
  Total samples:    ~1.5M
  Per-epoch avg:    ~27K
```

---

## 🚀 Quick Start Commands

```bash
# 1. Run training (one-time)
cd representation-learning/
bash src/scripts/run_train.sh

# 2. Run comparison (interactive)
bash src/scripts/run_comparison.sh notebook
# or (batch)
bash src/scripts/run_comparison.sh script

# 3. View results
cat results/comparison/logs/comparison.log
ls -lh results/comparison/*.png results/comparison/*.csv
```

---

## 📚 Related Files

| File | Purpose |
|------|---------|
| `src/engine/trainer.py` | M2M training loop (calls synthesizer) |
| `src/engine/evaluator.py` | Metrics computation (accuracy, balanced_acc, confusion matrix) |
| `src/data/cifar.py` | Dataset loading with imbalancing |
| `src/models/resnet.py` | ResNet18 architecture |
| `src/utils/config.py` | Config loading |

---

## ⚡ Performance Notes

- **Notebook mode:** Better for exploration, visualization, understanding
- **Script mode:** Faster for batch processing, easier to automate
- **Output size:** ~5-10 MB (3 PNG + logs)
- **Runtime:** ~2-5 minutes (depends on test set size)

---

## ❓ Common Questions

**Q: Do I need to run training first?**  
A: Yes. Comparison pipeline uses results from training (`history.json`, `best.pt`).

**Q: Can I modify the notebook?**  
A: Yes! Notebook is designed for exploration. Add custom cells, change plots, etc.

**Q: How to export results to report?**  
A: Notebook generates PNG files ready for presentations. CSVs for data analysis.

**Q: Can I run without Jupyter?**  
A: Yes! Use `bash src/scripts/run_comparison.sh script` for no-GUI mode.

---

## 📞 Support

- **Quick start:** See `COMPARISON_QUICKSTART.md`
- **Detailed docs:** See `COMPARISON_PIPELINE.md`
- **Code questions:** Check docstrings in `comparison.py`
- **Issues:** Check logs in `results/comparison/logs/comparison.log`

---

**Ready to analyze? Start with:** `bash src/scripts/run_comparison.sh notebook`
