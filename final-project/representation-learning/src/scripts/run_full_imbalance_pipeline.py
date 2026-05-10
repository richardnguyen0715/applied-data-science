#!/usr/bin/env python
"""End-to-end pipeline for baseline vs balanced imbalance comparison.

This script runs two explicit phases:
1) Train + evaluate without balancing (baseline)
2) Train + evaluate with balancing (M2M enabled)

It then exports:
- per-class counts before/after balancing
- sample visualizations before/after balancing
- evaluation comparison tables and plots
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from torch.optim import SGD

from src.data.cifar import create_cifar10_dataloaders
from src.engine.evaluator import Evaluator
from src.engine.trainer import M2MTrainer
from src.main import run_training
from src.models.resnet import build_resnet18
from src.utils.config import load_config
from src.utils.logger import setup_logger


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _denormalize(images: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return (images * std_t + mean_t).clamp(0.0, 1.0)


def _save_before_samples(
    train_loader: torch.utils.data.DataLoader[Any],
    out_path: Path,
    mean: List[float],
    std: List[float],
    num_samples: int,
) -> None:
    images, _ = next(iter(train_loader))
    images = images[:num_samples]
    images = _denormalize(images, mean, std)
    vutils.save_image(images, str(out_path), nrow=min(8, num_samples), pad_value=0.02)


def _save_after_samples_m2m(
    model: torch.nn.Module,
    trainer: M2MTrainer,
    train_loader: torch.utils.data.DataLoader[Any],
    out_path: Path,
    mean: List[float],
    std: List[float],
    num_pairs: int,
    num_classes: int,
    device: torch.device,
) -> None:
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    class_counts = torch.bincount(labels, minlength=num_classes)
    major_class = int(torch.argmax(class_counts).item())
    minor_class = int(torch.argmin(class_counts).item())

    source_indices = torch.where(labels == major_class)[0]
    if source_indices.numel() == 0:
        raise RuntimeError("Could not find majority class source samples.")

    use_pairs = min(num_pairs, int(source_indices.numel()))
    selected = source_indices[torch.randperm(source_indices.numel(), device=device)[:use_pairs]]

    source_images = images[selected].detach()
    target_labels = torch.full((use_pairs,), fill_value=minor_class, device=device, dtype=torch.long)
    synthesized_images = trainer.synthesizer.synthesize(model, source_images, target_labels)

    src_denorm = _denormalize(source_images, mean, std)
    synth_denorm = _denormalize(synthesized_images, mean, std)
    grid = torch.cat([src_denorm, synth_denorm], dim=0)
    vutils.save_image(grid, str(out_path), nrow=use_pairs, pad_value=0.02)


def _estimate_pre_post_distribution(
    trainer: M2MTrainer,
    train_loader: torch.utils.data.DataLoader[Any],
    num_classes: int,
    active_epoch: int,
    max_batches: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    pre_counts = torch.zeros(num_classes, dtype=torch.long)
    post_counts = torch.zeros(num_classes, dtype=torch.long)

    for batch_idx, (images, labels) in enumerate(train_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        pre_counts += torch.bincount(labels.cpu(), minlength=num_classes)
        _, aug_labels, _ = trainer._build_augmented_batch(images, labels, active_epoch)
        post_counts += torch.bincount(aug_labels.detach().cpu(), minlength=num_classes)

    return pre_counts.numpy(), post_counts.numpy()


def _prepare_phase_config(
    base_config: Dict[str, Any],
    output_dir: Path,
    phase: str,
    epochs_override: int,
    disable_balance: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["training"]["checkpoint_dir"] = str(output_dir / phase / "checkpoints")
    cfg["training"]["log_dir"] = str(output_dir / phase / "logs")

    if epochs_override > 0:
        cfg["training"]["epochs"] = int(epochs_override)

    if disable_balance:
        cfg["training"]["max_synth_per_batch"] = 0

    return cfg


def _build_dist_plot(df_counts: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(df_counts))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, df_counts["baseline_before"], width=width, label="Baseline (before)")
    ax.bar(x, df_counts["balanced_before"], width=width, label="Balanced run (before)")
    ax.bar(x + width, df_counts["balanced_after"], width=width, label="Balanced run (after M2M)")

    ax.set_xticks(x)
    ax.set_xticklabels(df_counts["class_name"], rotation=30)
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution Comparison: Before vs After Balancing")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_metrics_plot(df_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    baseline = df_metrics[df_metrics["phase"] == "baseline"].iloc[0]
    balanced = df_metrics[df_metrics["phase"] == "balanced"].iloc[0]

    ax.bar(x - width / 2, [baseline["accuracy"], baseline["balanced_accuracy"]], width=width, label="Baseline")
    ax.bar(x + width / 2, [balanced["accuracy"], balanced["balanced_accuracy"]], width=width, label="Balanced")

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Balanced Accuracy"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Evaluation Comparison: Before vs After Balancing")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_per_class_accuracy_plot(df_per_class: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(df_per_class))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, df_per_class["baseline_accuracy"], width=width, label="Baseline")
    ax.bar(x + width / 2, df_per_class["balanced_accuracy"], width=width, label="Balanced")

    ax.set_xticks(x)
    ax.set_xticklabels(df_per_class["class_name"], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Per-class accuracy")
    ax.set_title("Per-class Accuracy Before vs After Balancing")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("full_comparison_pipeline", str(output_dir / "logs"), "pipeline.log")
    logger.info("Starting full baseline-vs-balanced pipeline")

    base_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(base_config["dataset"].get("num_classes", 10))
    mean = list(base_config["dataset"].get("mean", [0.4914, 0.4822, 0.4465]))
    std = list(base_config["dataset"].get("std", [0.2023, 0.1994, 0.2010]))

    # Phase 1: baseline training + evaluation
    logger.info("Phase 1/2: baseline training and evaluation (no balancing)")
    baseline_cfg = _prepare_phase_config(
        base_config,
        output_dir,
        phase="baseline",
        epochs_override=args.baseline_epochs,
        disable_balance=True,
    )
    baseline_result = run_training(baseline_cfg)

    # Phase 2: balanced training + evaluation
    logger.info("Phase 2/2: balanced training and evaluation (M2M enabled)")
    balanced_cfg = _prepare_phase_config(
        base_config,
        output_dir,
        phase="balanced",
        epochs_override=args.balanced_epochs,
        disable_balance=False,
    )
    balanced_result = run_training(balanced_cfg)

    # Build loaders for distribution + visualization analysis
    baseline_train_loader, baseline_test_loader, baseline_dist = create_cifar10_dataloaders(baseline_cfg)
    balanced_train_loader, balanced_test_loader, balanced_dist = create_cifar10_dataloaders(balanced_cfg)

    # Load balanced best checkpoint for post-balance simulation and visualization
    balanced_ckpt_path = Path(balanced_cfg["training"]["checkpoint_dir"]) / "best.pt"
    balanced_model = build_resnet18(num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(balanced_ckpt_path, map_location=device)
    balanced_model.load_state_dict(ckpt["model_state_dict"])

    evaluator = Evaluator(num_classes=num_classes, device=device, show_progress=False)
    dummy_optimizer = SGD(balanced_model.parameters(), lr=0.0)
    analysis_trainer = M2MTrainer(
        model=balanced_model,
        optimizer=dummy_optimizer,
        train_loader=balanced_train_loader,
        val_loader=balanced_test_loader,
        evaluator=evaluator,
        config=balanced_cfg,
        device=device,
        logger=logger,
        scheduler=None,
    )

    pre_counts, post_counts = _estimate_pre_post_distribution(
        trainer=analysis_trainer,
        train_loader=balanced_train_loader,
        num_classes=num_classes,
        active_epoch=max(int(balanced_cfg["warmup_epochs"]), 1),
        max_batches=args.max_batches_distribution,
        device=device,
    )

    # Save visual samples before and after balancing
    visuals_dir = output_dir / "visualizations"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    _save_before_samples(
        balanced_train_loader,
        visuals_dir / "samples_before_balance.png",
        mean,
        std,
        args.num_visual_samples,
    )
    _save_after_samples_m2m(
        model=balanced_model,
        trainer=analysis_trainer,
        train_loader=balanced_train_loader,
        out_path=visuals_dir / "samples_after_balance_m2m.png",
        mean=mean,
        std=std,
        num_pairs=args.num_m2m_pairs,
        num_classes=num_classes,
        device=device,
    )

    # Save class distribution table
    df_counts = pd.DataFrame(
        {
            "class_id": list(range(num_classes)),
            "class_name": CIFAR10_CLASS_NAMES[:num_classes],
            "baseline_before": [baseline_dist.get(i, 0) for i in range(num_classes)],
            "balanced_before": [balanced_dist.get(i, 0) for i in range(num_classes)],
            "balanced_after": post_counts.tolist(),
        }
    )
    df_counts["after_vs_before_ratio"] = (
        df_counts["balanced_after"] / df_counts["balanced_before"].replace(0, np.nan)
    )
    df_counts.to_csv(output_dir / "class_distribution_comparison.csv", index=False)

    # Save metrics comparison table
    baseline_metrics = baseline_result["final_metrics"]
    balanced_metrics = balanced_result["final_metrics"]
    df_metrics = pd.DataFrame(
        [
            {
                "phase": "baseline",
                "accuracy": baseline_metrics["accuracy"],
                "balanced_accuracy": baseline_metrics["balanced_accuracy"],
            },
            {
                "phase": "balanced",
                "accuracy": balanced_metrics["accuracy"],
                "balanced_accuracy": balanced_metrics["balanced_accuracy"],
            },
        ]
    )
    df_metrics.to_csv(output_dir / "evaluation_comparison.csv", index=False)

    # Save per-class accuracy comparison table + plot
    baseline_per_class = baseline_metrics["per_class_accuracy"]
    balanced_per_class = balanced_metrics["per_class_accuracy"]
    df_per_class = pd.DataFrame(
        {
            "class_id": list(range(num_classes)),
            "class_name": CIFAR10_CLASS_NAMES[:num_classes],
            "baseline_accuracy": [float(baseline_per_class[i]) for i in range(num_classes)],
            "balanced_accuracy": [float(balanced_per_class[i]) for i in range(num_classes)],
        }
    )
    df_per_class["delta_accuracy"] = df_per_class["balanced_accuracy"] - df_per_class["baseline_accuracy"]
    df_per_class.to_csv(output_dir / "per_class_accuracy_comparison.csv", index=False)

    # Save plots
    _build_dist_plot(df_counts, output_dir / "class_distribution_comparison.png")
    _build_metrics_plot(df_metrics, output_dir / "evaluation_comparison.png")
    _build_per_class_accuracy_plot(df_per_class, output_dir / "per_class_accuracy_comparison.png")

    # Save lightweight JSON summary for automation
    summary = {
        "baseline": {
            "accuracy": float(baseline_metrics["accuracy"]),
            "balanced_accuracy": float(baseline_metrics["balanced_accuracy"]),
        },
        "balanced": {
            "accuracy": float(balanced_metrics["accuracy"]),
            "balanced_accuracy": float(balanced_metrics["balanced_accuracy"]),
        },
        "outputs": {
            "class_distribution_csv": str(output_dir / "class_distribution_comparison.csv"),
            "evaluation_csv": str(output_dir / "evaluation_comparison.csv"),
            "per_class_accuracy_csv": str(output_dir / "per_class_accuracy_comparison.csv"),
            "before_samples": str(visuals_dir / "samples_before_balance.png"),
            "after_samples": str(visuals_dir / "samples_after_balance_m2m.png"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    report_lines = [
        "# Full Imbalance Comparison Report",
        "",
        "## Training & Evaluation Phases",
        "1. Baseline: train + eval without balancing (`max_synth_per_batch=0`)",
        "2. Balanced: train + eval with M2M synthesis enabled",
        "",
        "## Final Metrics",
        f"- Baseline Accuracy: {baseline_metrics['accuracy']:.4f}",
        f"- Baseline Balanced Accuracy: {baseline_metrics['balanced_accuracy']:.4f}",
        f"- Balanced Accuracy: {balanced_metrics['accuracy']:.4f}",
        f"- Balanced Balanced Accuracy: {balanced_metrics['balanced_accuracy']:.4f}",
        "",
        "## Generated Files",
        "- class_distribution_comparison.csv",
        "- evaluation_comparison.csv",
        "- per_class_accuracy_comparison.csv",
        "- class_distribution_comparison.png",
        "- evaluation_comparison.png",
        "- per_class_accuracy_comparison.png",
        "- visualizations/samples_before_balance.png",
        "- visualizations/samples_after_balance_m2m.png",
        "",
        "## Notes",
        "- `balanced_after` counts are estimated from augmented batches (M2M active),",
        "  optionally limited by `--max-batches-distribution` for faster runs.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Pipeline completed. Outputs saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete baseline vs balanced comparison pipeline")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="results/full_imbalance_comparison")
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=0,
        help="Override baseline training epochs (0 = use config value)",
    )
    parser.add_argument(
        "--balanced-epochs",
        type=int,
        default=0,
        help="Override balanced training epochs (0 = use config value)",
    )
    parser.add_argument(
        "--max-batches-distribution",
        type=int,
        default=0,
        help="Max batches to estimate post-balance counts (0 = full epoch)",
    )
    parser.add_argument("--num-visual-samples", type=int, default=16)
    parser.add_argument("--num-m2m-pairs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
