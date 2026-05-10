# Full Imbalance Comparison Pipeline

Script: `src/scripts/run_full_imbalance_pipeline.py`

## Muc tieu

Pipeline nay tao 2 giai doan ro rang:

1. Baseline: train + eval KHONG balance (`max_synth_per_batch=0`)
2. Balanced: train + eval CO M2M balance

Va xuat day du:

1. So luong data moi class truoc/sau balance
2. Visualization mau truoc/sau balance
3. Ket qua danh gia truoc/sau balance

## Chay nhanh

```bash
cd final-project/representation-learning
/home/tgng/Coding/applied-data-science/.venv/bin/python3 -m src.scripts.run_full_imbalance_pipeline \
  --config src/configs/config.yaml \
  --output-dir results/full_imbalance_comparison \
  --baseline-epochs 1 \
  --balanced-epochs 1 \
  --max-batches-distribution 2
```

## Output chinh

- `results/full_imbalance_comparison/class_distribution_comparison.csv`
- `results/full_imbalance_comparison/evaluation_comparison.csv`
- `results/full_imbalance_comparison/class_distribution_comparison.png`
- `results/full_imbalance_comparison/evaluation_comparison.png`
- `results/full_imbalance_comparison/visualizations/samples_before_balance.png`
- `results/full_imbalance_comparison/visualizations/samples_after_balance_m2m.png`
- `results/full_imbalance_comparison/REPORT.md`
- `results/full_imbalance_comparison/summary.json`

## Ghi chu

- `balanced_after` la phan bo hieu dung tren augmented batches (M2M active), khong phai rewrite dataset goc tren dia.
- Neu muon tinh day du tren toan bo epoch augmented, bo `--max-batches-distribution`.
