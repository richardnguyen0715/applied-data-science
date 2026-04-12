"""Imbalance analysis and visualization utilities."""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def analyze_class_distribution(labels: List[int]) -> Dict[int, int]:
    """
    Analyze class distribution in a dataset.

    Args:
        labels: List of class labels.

    Returns:
        Dictionary mapping class indices to sample counts.
    """
    return dict(sorted(Counter(labels).items()))


def print_class_distribution(class_counts: Dict[int, int]) -> None:
    """
    Print class distribution in a formatted manner.

    Args:
        class_counts: Dictionary of class to sample count.
    """
    total = sum(class_counts.values())
    print("\nClass Distribution:")
    print("-" * 50)
    for class_idx, count in sorted(class_counts.items()):
        percentage = (count / total) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"Class {class_idx:2d}: {count:5d} ({percentage:5.1f}%) | {bar}")
    print("-" * 50)
    print(f"Total: {total} samples\n")


def get_minority_classes(
    class_counts: Dict[int, int],
    threshold_percentile: float = 50.0,
) -> List[int]:
    """
    Identify minority classes based on threshold.

    Args:
        class_counts: Dictionary of class to sample count.
        threshold_percentile: Percentile below which classes are minority.

    Returns:
        List of minority class indices.
    """
    counts = np.array(list(class_counts.values()))
    threshold = np.percentile(counts, threshold_percentile)
    minority = [
        cls_idx for cls_idx, count in class_counts.items() if count < threshold
    ]
    return minority


def get_imbalance_ratio(class_counts: Dict[int, int]) -> float:
    """
    Calculate the imbalance ratio (max/min class samples).

    Args:
        class_counts: Dictionary of class to sample count.

    Returns:
        Imbalance ratio.
    """
    counts = list(class_counts.values())
    return max(counts) / min(counts)


def plot_class_distribution(
    class_counts: Dict[int, int],
    title: str = "Class Distribution",
    save_path: str = None,
) -> None:
    """
    Plot class distribution as a bar chart.

    Args:
        class_counts: Dictionary of class to sample count.
        title: Title of the plot.
        save_path: Optional path to save the figure.
    """
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color="steelblue", edgecolor="black")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(classes)
    plt.grid(axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()
