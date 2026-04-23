"""Visualization utilities."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Plot training curves.

    Args:
        history: Dictionary with training history.
        save_path: Path to save figure.
        show: Whether to show plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix (num_classes, num_classes).
        class_names: List of class names.
        save_path: Path to save figure.
        show: Whether to show plot.
    """
    num_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    class_counts: Dict[int, int],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Plot class distribution.

    Args:
        class_counts: Dictionary with class -> count mapping.
        save_path: Path to save figure.
        show: Whether to show plot.
    """
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color="steelblue")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(classes)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
