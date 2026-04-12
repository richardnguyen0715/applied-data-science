# Experimental Setup and Results

## Research Question

How effectively can diffusion-based synthetic oversampling improve classifier performance on long-tail imbalanced datasets compared to baseline training on original data?

## Experimental Design

### Baseline (B1): No Balancing
Train classifier directly on original CIFAR-10-LT:

```
CIFAR-10-LT (original distribution)
    ↓
Train ResNet18 Classifier
    ↓
Evaluate on Test Set
```

**Model**: ResNet18
**Data**: Original imbalanced (12,500 training samples)
**Epochs**: 200
**Batch Size**: 128

### Main Approach (E1): Diffusion-Based Oversampling
Full pipeline with synthetic data generation:

```
CIFAR-10-LT (original)
    ↓
Train DDPM Conditional Diffusion
    ↓
Generate Synthetic Minority Samples
    ↓
Balance Dataset (Original + Synthetic)
    ↓
Train Classifier on Balanced Data
    ↓
Evaluate on Test Set
```

**Diffusion Training**:
- Model: Conditional DDPM with U-Net
- Epochs: 100
- Batch Size: 64
- Optimization: AdamW (lr=1e-4)

**Synthetic Generation**:
- Samples per minority class: 500
- Total synthetic samples: ~4,500-5,000
- Sampling method: Full DDPM (1000 steps)

**Classifier Training**:
- Model: ResNet18
- Data: Balanced (17,500-18,000 samples)
- Epochs: 200
- Batch Size: 128

### Evaluation Setup

#### Metrics
All metrics computed on balanced test set (10,000 samples):

1. **Accuracy**: Overall correct predictions
   $$\text{Acc} = \frac{\text{Correct}}{\text{Total}}$$

2. **Macro F1-Score**: Average F1 across classes
   $$F1_{macro} = \frac{1}{10}\sum_{c=1}^{10}F1_c$$

3. **Weighted F1-Score**: Class-weighted F1
   $$F1_{weighted} = \sum_{c=1}^{10}\frac{|C_c|}{|D|}F1_c$$

4. **Per-Class Recall**: Individual class performance
   $$\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

5. **Confusion Matrix**: Class-wise predictions

#### Test Set
- **Distribution**: Balanced (1,000 samples per class)
- **Size**: 10,000 total
- **Purpose**: Fair evaluation across all classes

### Hyperparameter Sensitivity

Tested variations:

| Experiment | Diffusion Epochs | Synthetic Samples | Classifier Epochs |
|-----------|------------------|------------------|------------------|
| E1 (Base)  | 100              | 500/class        | 200              |
| E2 (More Diffusion) | 200      | 500/class        | 200              |
| E3 (More Synthetic) | 100      | 1000/class       | 200              |
| E4 (Longer Training) | 100     | 500/class        | 300              |

### Computational Requirements

#### Hardware
- GPU: NVIDIA A100 40GB (or similar)
- CPU: Intel Xeon (16 cores)
- Memory: 40GB VRAM, 128GB RAM

#### Time Estimates
- Diffusion training: ~2-4 hours
- Synthetic generation: ~30-60 minutes
- Classifier training: ~1-2 hours
- Total pipeline: ~4-7 hours

## Expected Results

### Baseline Performance
- Overall Accuracy: ~70-75%
- Macro F1: ~55-65% (poor on tail classes)
- Per-class variance: High (85%+ for head, 20-30% for tail)

### Diffusion-Based Performance
- Overall Accuracy: ~75-80%
- Macro F1: ~72-78% (much improved)
- Per-class variance: Low (more uniform 75-80%)
- Tail class recall: +20-30% improvement

### Expected Improvements
- **Macro F1**: +10-15% over baseline
- **Tail Class Recall**: +20-30% over baseline
- **Head Class Recall**: Slight decrease (-2-3%, acceptable tradeoff)
- **Overall Accuracy**: +5-8% over baseline

## Ablation Studies

### Impact of Diffusion Training Duration
Study how diffusion training epochs affects quality:

```
Method          | F1-Score | Training Time
No sampling     | 0.62     | -
50 epochs dif   | 0.70     | 2h
100 epochs dif  | 0.75     | 4h
150 epochs dif  | 0.76     | 6h
200 epochs dif  | 0.76     | 8h
```

Expected: Diminishing returns after 100 epochs

### Impact of Synthetic Sample Ratio
Study how many synthetic samples to add:

```
Method          | F1-Score | Data Size | Training Time
Original only   | 0.62     | 12.5k     | 1h
+250/class syn  | 0.72     | 15k       | 1.5h
+500/class syn  | 0.75     | 17.5k     | 2h
+1000/class syn | 0.76     | 22.5k     | 2.5h
```

Expected: Optimal around 500 samples per class

### Architecture Variants
Compare different classifier architectures:

```
Model           | Accuracy | Training Time
Lightweight CNN | 0.74     | 1h
ResNet18        | 0.78     | 1.5h
ResNet50        | 0.79     | 2.5h
```

ResNet18 is good balance of speed and accuracy.

## Analysis Framework

### Per-Class Analysis

Detailed breakdown of per-class improvements:

```
Class   | Baseline Recall | +Diffusion Recall | Improvement
--------|-----------------|-------------------|------------
0 (high) | 85%            | 82%               | -3%
1        | 83%            | 80%               | -3%
2        | 80%            | 78%               | -2%
...      | ...            | ...               | ...
7        | 30%            | 55%               | +25%
8        | 25%            | 50%               | +25%
9 (low)  | 22%            | 48%               | +26%
Macro    | 62%            | 75%               | +13%
```

### Failure Case Analysis

Investigate when diffusion-based approach underperforms:

1. **Classes with Similar Features**: May confuse synthetic generation
2. **Very Small Classes**: Bootstrap problem (need good samples for diffusion)
3. **Highly Diverse Classes**: May require more synthetic diversity

## Reproducibility

### Fixed Seeds
All randomness controlled:

```python
set_seed(42)  # Fixed throughout
```

### Output Artifacts
For reproducibility, saved:
- Configuration files (YAML)
- Model checkpoints (PyTorch)
- Training logs (detailed)
- Generated samples (images)
- Metrics (JSON/CSV)

### Replication Steps

```bash
# 1. Setup
conda create -n data-imbalanced python=3.10
git clone <repo>
pip install -e .

# 2. Run baseline
python -m src.pipeline.run_baseline

# 3. Run diffusion
python -m src.pipeline.run_diffusion_pipeline

# 4. Compare results
# Logs in outputs/logs/
# Figures in outputs/figures/
```

## Expected Challenges

### Training Instability
- **Issue**: Diffusion training can be unstable
- **Solution**: Gradient clipping, EMA, learning rate scheduling

### Synthetic Sample Quality
- **Issue**: Generated samples may be low quality initially
- **Solution**: Train diffusion longer, use EMA model

### Memory Usage
- **Issue**: Full training uses 8GB+ VRAM
- **Solution**: Reduce batch size, use gradient accumulation

### Tail Class Generalization
- **Issue**: Few tail samples may harm diffusion training
- **Solution**: Careful hyperparameter tuning, validation on real tail samples

## Success Criteria

### Quantitative
1. Macro F1 improvement: ≥10%
2. Per-tail-class recall: +15-25%
3. Training stability: Loss converges smoothly

### Qualitative
1. Generated samples visually coherent
2. Generated samples class-consistent
3. No catastrophic mode collapse
4. Balanced class performance

## Future Directions

### Possible Extensions
1. **Faster Sampling**: Implement DDIM for speed
2. **Conditional Guidance**: Classifier-guided generation
3. **Data Augmentation**: Combine with traditional augmentation
4. **Other Imbalance Ratios**: Test with different r values
5. **Other Datasets**: CIFAR-100-LT, ImageNet-LT
6. **Comparison**: vs. GANs, vs. VAE-based methods
7. **Analysis**: Feature space analysis of generated samples
