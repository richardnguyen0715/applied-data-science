# Imbalance Handling Comparison Pipeline

Pipeline để so sánh kết quả trước và sau khi xử lý class imbalancing thông qua M2M (Major-to-Minor) synthesis.

## 📋 Tổng Quan

Dự án này xử lý vấn đề **class imbalancing** trong CIFAR-10 dataset bằng:
- **Baseline**: Standard training trên data imbalanced (class `dog` có 50 samples, `airplane` có 5000 samples)
- **M2M Synthesis**: Tạo synthetic samples từ majority classes để balance dataset

## 🚀 Cách Sử Dụng

### 1. Jupyter Notebook (Recommended)

Notebook tương tác với đầy đủ visualizations:

```bash
cd notebooks
jupyter notebook comparison_pipeline.ipynb
```

**Nội dung notebook:**
- 📊 Load training history và visualize learning curves
- 🔍 Evaluate best model trên test set
- 📈 Per-class performance analysis
- 🔄 M2M synthesis impact assessment
- 📉 Confusion matrix và error analysis
- 📊 Summary report với metrics chi tiết

### 2. Command Line Script

Chạy comparison pipeline từ command line:

```bash
# Run with default paths
python src/scripts/run_comparison.py

# Run with custom paths
python src/scripts/run_comparison.py \
  --config src/configs/config.yaml \
  --results-dir results/cifar10 \
  --output-dir results/comparison
```

**Output:**
- CSV files: Per-class metrics, overall metrics
- PNG files: Training curves, per-class accuracy, imbalance analysis
- Log file: Detailed comparison report

## 📊 Main Metrics

### 1. **Standard Accuracy**
Tỷ lệ predictions đúng trên toàn bộ test set. **Không fair** cho imbalanced data vì bị dominated bởi majority classes.

```
Accuracy = (TP + TN) / Total
```

### 2. **Balanced Accuracy** ⭐ (Chính xác hơn cho imbalanced data)
Trung bình accuracy của từng class. Cho mỗi class cân nặng bằng nhau.

```
Balanced Accuracy = (1/C) * Σ(TP_i / (TP_i + FN_i))
```

### 3. **Per-Class Accuracy**
Độ chính xác riêng biệt cho mỗi class. Giúp xác định classes nào model yếu.

### 4. **Improvement Metrics**
- Accuracy improvement từ M2M synthesis
- Performance gap giữa minority vs majority classes
- Variance giảm giữa per-class accuracies

## 📁 Output Structure

```
results/comparison/
├── logs/
│   └── comparison.log
├── per_class_performance.csv      # Accuracy, train count, train %
├── overall_metrics.csv            # Summary statistics
├── training_curves.png            # Loss, accuracy, balanced acc, synth activity
├── per_class_accuracy.png         # Per-class accuracy bars với references
└── imbalance_vs_accuracy.png      # Training distribution vs test accuracy correlation
```

## 🔍 Cách Đọc Kết Quả

### Training Curves Plot

| Subplot | Ý Nghĩa |
|---------|---------|
| Loss | Giảm dần cho cả train/val → model converging |
| Standard Accuracy | Cao nhưng có thể bị bias bởi majority classes |
| Balanced Accuracy | Chính xác hơn cho imbalanced data, tăng sau warmup |
| M2M Activity | Số samples synthesized/epoch, 0 trong warmup epochs |

### Per-Class Accuracy

- **Bars**: Accuracy của từng class trên test set
- **Blue dashed line**: Standard accuracy (overall)
- **Red dashed line**: Balanced accuracy (fair average)
- **Insight**: Classes với ít training samples thường có accuracy thấp hơn

### Imbalance vs Accuracy

- **X-axis**: Classes sắp xếp theo training frequency (ít → nhiều)
- **Bar**: Số training samples
- **Line**: Test accuracy
- **Pattern**: Correlation giữa training samples và test accuracy

## 💡 Insights Từ Comparison

### Trước M2M (Warmup Period, Epochs 1-5)
- Model học trên imbalanced data
- Minority classes như `dog` có very low accuracy
- Model overfits trên majority classes
- Balanced accuracy thấp

### Sau M2M (Epochs 6+)
- Synthetic samples được tạo cho minority classes
- Per-class accuracy trở nên balanced hơn
- Balanced accuracy tăng đáng kể
- Performance gap giữa classes giảm

## 📈 Expected Results

Trên CIFAR-10 với imbalance factor 0.01:

| Metric | Trước M2M | Sau M2M | Improvement |
|--------|-----------|---------|-------------|
| Standard Accuracy | ~65% | ~68% | +3% |
| Balanced Accuracy | ~35% | ~58% | +23% |
| Std Dev (per-class) | High | Low | Variance ↓ |
| Worst Class Accuracy | ~5% | ~40% | +35% |

## 🛠️ Customization

### Thay đổi Evaluation Config

Edit `src/configs/config.yaml`:

```yaml
training:
  epochs: 60
  warmup_epochs: 5  # Khi nào M2M starts
  max_synth_per_batch: 32  # Số samples synthesized/batch

imbalance:
  imbalance_factor: 0.01  # Minority/Majority ratio
```

### Thay đổi Output Directory

```bash
python src/scripts/run_comparison.py \
  --output-dir custom_results/
```

## 🔗 Files Liên Quan

| File | Purpose |
|------|---------|
| `src/engine/trainer.py` | M2M training logic |
| `src/engine/evaluator.py` | Evaluation metrics |
| `src/utils/comparison.py` | Comparison utilities |
| `src/data/cifar.py` | Dataset imbalancing |
| `notebooks/comparison_pipeline.ipynb` | Interactive analysis |

## 📚 Further Reading

- Balanced Accuracy: https://en.wikipedia.org/wiki/Precision_and_recall#Balanced_accuracy
- M2M Synthesis: Major-to-Minor class augmentation
- Class Imbalance Handling: https://imbalanced-learn.org/

## ❓ FAQ

**Q: Tại sao lại dùng Balanced Accuracy thay vì Standard Accuracy?**  
A: Với imbalanced data, standard accuracy bị bias bởi majority classes. Balanced accuracy cân nặng bằng nhau cho mỗi class.

**Q: M2M có tạo samples hoàn toàn mới không?**  
A: Không, M2M synthesis sử dụng model gradients để biến đổi majority class samples thành minority classes.

**Q: Có thể so sánh với baseline (không có M2M) không?**  
A: Có thể train model mà skip M2M phase (set `max_synth_per_batch: 0`), rồi so sánh results.

**Q: Làm sao biết M2M có hiệu quả không?**  
A: So sánh balanced accuracy trước/sau M2M activation (between warmup and post-warmup epochs).
