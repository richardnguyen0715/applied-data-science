"""Comparison utilities for imbalance handling analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from engine.evaluator import Evaluator
from models.resnet import build_resnet18


class ImbalanceComparisonAnalyzer:
    """Analyze and compare model performance before and after imbalance handling."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        results_dir: Path,
    ) -> None:
        """Initialize analyzer.

        Args:
            config: Training configuration.
            device: Compute device.
            results_dir: Directory containing results and checkpoints.
        """
        self.config = config
        self.device = device
        self.results_dir = Path(results_dir)
        self.num_classes = int(config["dataset"]["num_classes"])

        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def load_training_history(self) -> Tuple[pd.DataFrame, bool]:
        """Load training history from JSON.

        Returns:
            Tuple of (history DataFrame, success flag).
        """
        history_path = self.results_dir / "logs" / "history.json"

        if not history_path.exists():
            return pd.DataFrame(), False

        with open(history_path, "r") as f:
            history = json.load(f)

        df = pd.DataFrame(history)
        return df, True

    def load_checkpoint(self, checkpoint_name: str = "best.pt") -> Tuple[torch.nn.Module, Dict[str, Any], bool]:
        """Load model checkpoint.

        Args:
            checkpoint_name: Name of checkpoint file.

        Returns:
            Tuple of (model, checkpoint dict, success flag).
        """
        model = build_resnet18(num_classes=self.num_classes, pretrained=False).to(self.device)
        checkpoint_path = self.results_dir / "checkpoints" / checkpoint_name

        if not checkpoint_path.exists():
            return model, {}, False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint, True

    def evaluate_and_compare(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        train_dist: Dict[int, int],
    ) -> Dict[str, Any]:
        """Evaluate model and generate comparison metrics.

        Args:
            model: Trained model.
            test_loader: Test dataloader.
            train_dist: Training class distribution.

        Returns:
            Dictionary with analysis results.
        """
        evaluator = Evaluator(num_classes=self.num_classes, device=self.device, show_progress=True)
        test_metrics = evaluator.evaluate(model, test_loader, progress_desc="Evaluating model")

        per_class_acc = test_metrics["per_class_accuracy"]
        class_counts = list(train_dist.values())

        return {
            "test_metrics": test_metrics,
            "per_class_accuracy": per_class_acc,
            "class_counts": class_counts,
            "class_names": self.class_names,
        }

    def generate_summary_report(
        self,
        history_df: pd.DataFrame,
        comparison_results: Dict[str, Any],
        train_dist: Dict[int, int],
    ) -> str:
        """Generate comprehensive comparison report.

        Args:
            history_df: Training history.
            comparison_results: Results from evaluate_and_compare.
            train_dist: Training class distribution.

        Returns:
            Formatted report string.
        """
        test_metrics = comparison_results["test_metrics"]
        per_class_acc = comparison_results["per_class_accuracy"]
        class_counts = comparison_results["class_counts"]

        report = []
        report.append("\n" + "=" * 80)
        report.append("IMBALANCE HANDLING COMPARISON REPORT: Before vs After M2M Synthesis")
        report.append("=" * 80)

        # Dataset imbalance
        report.append("\n📊 DATASET IMBALANCE:")
        report.append(f"  Imbalance Factor: {self.config['imbalance']['imbalance_factor']}")
        report.append(f"  Majority Class ({self.class_names[0]}): {class_counts[0]} samples")
        report.append(f"  Minority Class ({self.class_names[5]}): {class_counts[5]} samples")
        report.append(f"  Ratio (Minority/Majority): {class_counts[5] / class_counts[0]:.2%}")

        # Performance metrics
        report.append("\n📈 PERFORMANCE METRICS (Test Set):")
        report.append(f"  Standard Accuracy: {test_metrics['accuracy']:.4f}")
        report.append(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        improvement = (test_metrics["balanced_accuracy"] - test_metrics["accuracy"]) * 100
        report.append(f"  Accuracy Improvement: {improvement:+.2f}%")

        # Per-class performance
        report.append("\n🎯 PER-CLASS PERFORMANCE:")
        best_idx = np.argmax(per_class_acc)
        worst_idx = np.argmin(per_class_acc)
        report.append(f"  Best: {self.class_names[best_idx]} ({max(per_class_acc):.4f})")
        report.append(f"  Worst: {self.class_names[worst_idx]} ({min(per_class_acc):.4f})")
        report.append(f"  Std Dev: {np.std(per_class_acc):.4f}")

        # M2M synthesis impact
        if not history_df.empty:
            warmup_epochs = int(self.config["warmup_epochs"])
            df_after_warmup = history_df[history_df["epoch"] > warmup_epochs].copy()

            report.append("\n🔄 M2M SYNTHESIS CONTRIBUTION:")
            report.append(f"  M2M Active Epochs: {len(df_after_warmup)}")
            total_synthesized = df_after_warmup["num_synthesized"].sum()
            report.append(f"  Total Augmented Samples: {total_synthesized:,.0f}")
            report.append(f"  Avg per Epoch: {total_synthesized / len(df_after_warmup):,.0f}")

            warmup_metrics = history_df[history_df["epoch"] <= warmup_epochs].iloc[-1]
            final_metrics = history_df.iloc[-1]
            bal_acc_improvement = final_metrics["val_balanced_accuracy"] - warmup_metrics["val_balanced_accuracy"]
            report.append(f"  Balanced Acc Improvement: {bal_acc_improvement:+.4f}")

        report.append("\n" + "=" * 80 + "\n")
        return "\n".join(report)

    def save_analysis_results(
        self,
        comparison_results: Dict[str, Any],
        output_dir: Path | None = None,
    ) -> None:
        """Save analysis results to CSV and plots.

        Args:
            comparison_results: Results from evaluate_and_compare.
            output_dir: Output directory for files. Defaults to results_dir.
        """
        if output_dir is None:
            output_dir = self.results_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        per_class_acc = comparison_results["per_class_accuracy"]
        class_counts = comparison_results["class_counts"]

        # Per-class metrics CSV
        df_class = pd.DataFrame({
            "Class": self.class_names,
            "Accuracy": per_class_acc,
            "Train Count": class_counts,
            "Train %": [100 * c / sum(class_counts) for c in class_counts],
        })
        df_class.to_csv(output_dir / "per_class_performance.csv", index=False)

        # Overall metrics CSV
        test_metrics = comparison_results["test_metrics"]
        df_overall = pd.DataFrame({
            "metric": ["Accuracy", "Balanced Accuracy", "Std Dev", "Best Class", "Worst Class"],
            "value": [
                test_metrics["accuracy"],
                test_metrics["balanced_accuracy"],
                np.std(per_class_acc),
                max(per_class_acc),
                min(per_class_acc),
            ],
        })
        df_overall.to_csv(output_dir / "overall_metrics.csv", index=False)


def create_comparison_plots(
    history_df: pd.DataFrame,
    per_class_acc: List[float],
    class_names: List[str],
    class_counts: List[int],
    test_metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create comprehensive comparison visualizations.

    Args:
        history_df: Training history DataFrame.
        per_class_acc: Per-class accuracy list.
        class_names: Class names.
        class_counts: Training sample counts per class.
        test_metrics: Overall test metrics.
        output_dir: Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    if not history_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], "b-", linewidth=2, label="Train")
        axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], "r-", linewidth=2, label="Validation")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history_df["epoch"], history_df["train_accuracy"], "b-", linewidth=2, label="Train")
        axes[0, 1].plot(history_df["epoch"], history_df["val_accuracy"], "r-", linewidth=2, label="Validation")
        axes[0, 1].set_title("Standard Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(history_df["epoch"], history_df["val_balanced_accuracy"], "g-", linewidth=2)
        axes[1, 0].set_title("Balanced Accuracy (Metric for Imbalanced Data)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Balanced Accuracy")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].bar(history_df["epoch"], history_df["num_synthesized"], color="purple", alpha=0.7)
        axes[1, 1].set_title("M2M Synthesis Activity")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Synthesized Samples")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Per-class performance
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(class_names)))
    ax.barh(class_names, per_class_acc, color=colors)
    ax.axvline(test_metrics["accuracy"], color="blue", linestyle="--", linewidth=2,
              label=f"Standard: {test_metrics['accuracy']:.4f}")
    ax.axvline(test_metrics["balanced_accuracy"], color="red", linestyle="--", linewidth=2,
              label=f"Balanced: {test_metrics['balanced_accuracy']:.4f}")
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Per-Class Performance After M2M")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Imbalance vs accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_idx = np.argsort(class_counts)
    sorted_names = [class_names[i] for i in sorted_idx]
    sorted_counts = [class_counts[i] for i in sorted_idx]
    sorted_acc = [per_class_acc[i] for i in sorted_idx]

    ax2 = ax.twinx()
    bars = ax.bar(range(len(class_names)), sorted_counts, alpha=0.6, color="tab:blue", label="Train Samples")
    line = ax2.plot(range(len(class_names)), sorted_acc, "o-", color="tab:orange", linewidth=2,
                   markersize=8, label="Test Accuracy")

    ax.set_xlabel("Class (sorted by frequency)")
    ax.set_ylabel("Training Samples", color="tab:blue")
    ax2.set_ylabel("Test Accuracy", color="tab:orange")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title("Training Imbalance vs Test Accuracy")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "imbalance_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Plots saved to {output_dir}")
