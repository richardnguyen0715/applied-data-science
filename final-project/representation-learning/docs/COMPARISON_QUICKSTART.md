# Comparison Pipeline - Quick Start Guide

## 🎯 Mục Đích

Pipeline này so sánh hiệu quả xử lý **class imbalancing** trong CIFAR-10 dataset bằng **M2M (Major-to-Minor) synthesis**.

**Before vs After:**
- **Before**: Standard training trên imbalanced CIFAR-10 (minority class: 50 samples, majority: 5000 samples)
- **After**: Training với M2M synthesis augmentation (tạo synthetic minority samples)

**Kỳ vọng:**
- Balanced accuracy: 35% → 58% (+23%)
- Worst class accuracy: 5% → 40% (+35%)
- Performance variance giảm

---

## 📦 Prerequisites

1. **Training must be completed** - Tạo `results/cifar10/logs/history.json` và `results/cifar10/checkpoints/best.pt`
2. **Python 3.11+** - Với PyTorch, torchvision, pandas, matplotlib, seaborn

```bash
# Cài dependencies
pip install torch torchvision jupyter pandas matplotlib seaborn

# Hoặc dùng environment hiện tại
conda activate appliedDataScience
```

---

## 🚀 Cách Sử Dụng

### Option 1: Interactive Jupyter Notebook ⭐ (Recommended)

**Best for:** Exploration, understanding results, generating custom plots

```bash
# Cách 1a: Run script wrapper
bash src/scripts/run_comparison.sh notebook

# Cách 1b: Run directly
jupyter notebook notebooks/comparison_pipeline.ipynb

# Cách 1c: Run in VS Code
# Open notebooks/comparison_pipeline.ipynb in VS Code
```

**Notebook structure:**
1. Load libraries & config
2. Create dataloaders with class distribution
3. Load training history
4. Load best checkpoint & evaluate
5. Visualize training curves
6. Per-class performance analysis
7. Generate confusion matrix
8. Imbalance impact analysis
9. M2M synthesis contribution
10. Summary report
11. Export results to CSV/PNG

### Option 2: Command Line Script

**Best for:** Automation, batch processing, server environments

```bash
# Cách 2a: Run script wrapper
bash src/scripts/run_comparison.sh script

# Cách 2b: Run Python directly
python src/scripts/run_comparison.py

# Cách 2c: Custom paths
python src/scripts/run_comparison.py \
  --config src/configs/config.yaml \
  --results-dir results/cifar10 \
  --output-dir results/comparison
```

### Option 3: View Results

```bash
# View latest comparison
bash src/scripts/run_comparison.sh view-results

# Or manually
cat results/comparison/logs/comparison.log
```

---

## 📊 Output Files

### CSV Files (In `results/comparison/`)

**1. `per_class_performance.csv`**
```
Class,Accuracy,Train Count,Train %
airplane,0.8234,5000,25.00
automobile,0.7654,4000,20.00
...
dog,0.4123,50,0.25
```

**2. `overall_metrics.csv`**
```
metric,value
Accuracy,0.6823
Balanced Accuracy,0.5842
Std Dev,0.1245
Best Class,0.8234
Worst Class,0.4123
```

### PNG Files (Visualizations)

**1. `training_curves.png`** (2x2 grid)
- Top-left: Loss (train vs validation)
- Top-right: Standard accuracy (train vs validation)
- Bottom-left: Balanced accuracy (validation only)
- Bottom-right: M2M synthesis activity (samples per epoch)

**2. `per_class_accuracy.png`**
- Horizontal bars: Per-class accuracy
- Blue dashed line: Standard accuracy
- Red dashed line: Balanced accuracy
- Shows: Which classes model struggles with

**3. `imbalance_vs_accuracy.png`**
- X-axis: Classes (sorted by training frequency: minority→majority)
- Blue bars: Training samples per class
- Orange line: Test accuracy per class
- Shows: Correlation between imbalance and model performance

### Log Files

`logs/comparison.log` - Detailed comparison report:
- Dataset imbalance statistics
- Per-class performance metrics
- M2M synthesis contribution analysis
- Before/after comparison summary

---

## 📈 How to Read Results

### Key Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness. **Biased** for imbalanced data. |
| **Balanced Accuracy** ⭐ | (1/C) × Σ(TP_i / (TP_i+FN_i)) | Fair average per-class. **Best for imbalanced data.** |
| **Per-Class Acc** | TP_i / (TP_i + FN_i) | Individual class performance. |
| **Std Dev** | σ(per_class_acc) | Performance variance. Lower = more balanced. |

### Interpretation Example

```
Dataset Imbalance:
  Majority (airplane): 5000 samples
  Minority (dog): 50 samples
  Ratio: 1:100

Before M2M (Epoch 5):
  Accuracy: 0.6523          ← Looks good but biased
  Balanced Accuracy: 0.3421 ← Reveals true weakness
  Worst class (dog): 0.05   ← Model barely learns minority class

After M2M (Epoch 60):
  Accuracy: 0.6823          ← Improved slightly
  Balanced Accuracy: 0.5842 ← Much better! +23%
  Worst class (dog): 0.41   ← Significantly improved! +36%
  Std Dev: 0.0845           ← More balanced across classes
```

### What Good Results Look Like

✅ **Good signs:**
- Balanced accuracy > 50% (or halfway between random 10% and best possible)
- Balanced accuracy increase after warmup (M2M activation)
- Per-class accuracy more uniform (low std dev)
- Worst class accuracy >> random (10% for 10-class problem)

⚠️ **Warning signs:**
- Balanced accuracy barely changes after M2M activation
- Few per-class accuracies below overall accuracy
- High variance in per-class performance

---

## 🔍 Detailed Analysis Steps

### Step 1: Check Training History

Open notebook, cell 3 - Load and inspect training metrics:
- Loss should decrease monotonically
- Accuracy should increase overall
- Balanced accuracy should increase after epoch 5 (warmup end)

### Step 2: Evaluate Model

Cell 4 - Load checkpoint and evaluate:
- Accuracy: Standard metric
- Balanced accuracy: Fairness metric
- Per-class breakdown: Which classes suffer

### Step 3: Visual Analysis

Cells 5-6 - Training curves and per-class performance:
- Identify when model stops improving
- Spot which classes remain problematic
- Verify M2M is active (non-zero synthesis)

### Step 4: Imbalance Analysis

Cell 8 - Understand dataset impact:
- Compare minority vs majority class performance
- Verify correlation: fewer samples → lower accuracy
- Quantify performance gap

### Step 5: M2M Contribution

Cell 9 - M2M effectiveness:
- Active epochs: How many epochs did M2M run?
- Total samples: How many synthetic samples created?
- Accuracy improvement: How much did balanced accuracy improve?

---

## 💾 Saving Custom Analysis

In Jupyter notebook, add new cells after cell 11:

```python
# Custom analysis: Compare with baseline
import json

# Load baseline (if exists)
baseline_results = json.load(open('baseline_results.json'))

# Compare
improvement = {
    'accuracy_gain': test_metrics['accuracy'] - baseline_results['accuracy'],
    'balanced_acc_gain': test_metrics['balanced_accuracy'] - baseline_results['balanced_accuracy'],
    'per_class_improvement': [
        test_metrics['per_class_accuracy'][i] - baseline_results['per_class_accuracy'][i]
        for i in range(10)
    ]
}

print("Improvement over baseline:")
print(json.dumps(improvement, indent=2))
```

---

## 🛠️ Troubleshooting

### Q: "Dataset not found" error
**A:** Run training first to generate history.json and checkpoints:
```bash
bash src/scripts/run_train.sh
```

### Q: Jupyter not found
**A:** Install Jupyter:
```bash
pip install jupyter
```

### Q: Results look unchanged before/after M2M
**A:** Check:
1. M2M actually activated (check num_synthesized in history)
2. Warmup epochs configured correctly
3. Model converged (final val loss decreasing?)

### Q: Can't run shell scripts on Windows
**A:** Use Python directly:
```bash
python src/scripts/run_comparison.py
```

---

## 📚 Next Steps

1. **Save results:** Export CSVs and PNGs from notebook
2. **Compare baselines:** Run without M2M, compare results
3. **Tune parameters:** Adjust `imbalance_factor`, `max_synth_per_batch`
4. **Publish:** Use visualizations in presentations/papers

---

## 📖 Full Documentation

See [COMPARISON_PIPELINE.md](COMPARISON_PIPELINE.md) for:
- Detailed metrics explanation
- Customization options
- FAQ
- Further reading

---

## 📞 Quick Commands Reference

```bash
# Start interactive notebook
bash src/scripts/run_comparison.sh notebook

# Run as script
bash src/scripts/run_comparison.sh script

# View results
bash src/scripts/run_comparison.sh view-results

# Run training first (if needed)
bash src/scripts/run_train.sh

# Access results directory
cd results/comparison
ls *.csv *.png
```

---

**Ready? Start with:** `bash src/scripts/run_comparison.sh notebook`
