"""Visualization utilities."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from src.data.transforms import denormalize


def plot_class_distribution(
    targets: np.ndarray,
    title: str = "Class Distribution",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot class distribution.

    Args:
        targets: Target labels.
        title: Plot title.
        output_path: Path to save plot.
    """
    unique, counts = np.unique(targets, return_counts=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(unique, counts, color='steelblue', edgecolor='black')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: Optional[list] = None,
    title: str = "Training Curves",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot training curves.

    Args:
        train_losses: Training losses.
        val_losses: Validation losses.
        title: Plot title.
        output_path: Path to save plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, label='Train Loss', linewidth=2, color='steelblue')
    if val_losses is not None:
        ax.plot(val_losses, label='Val Loss', linewidth=2, color='coral')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Optional[list] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix.
        classes: List of class names.
        normalize: Whether to normalize.
        title: Plot title.
        output_path: Path to save plot.
    """
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm_display, cmap='Blues', aspect='auto')

    # Ticks
    num_classes = cm.shape[0]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))

    if classes is not None:
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)
    else:
        ax.set_xticklabels(range(num_classes), rotation=45)
        ax.set_yticklabels(range(num_classes))

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.close()


def plot_generated_images(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Generated Images",
    num_images: int = 16,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot generated images.

    Args:
        images: Generated images (B, 3, 32, 32) in range [0, 1].
        labels: Image labels.
        title: Plot title.
        num_images: Number of images to display.
        output_path: Path to save plot.
    """
    num_images = min(num_images, len(images))
    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    for idx in range(num_images):
        ax = axes[idx]
        img = images[idx]

        # Convert to (H, W, C) format
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        ax.imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
        ax.axis('off')

        if labels is not None:
            ax.set_title(f"Class {labels[idx]}", fontsize=10)

    # Hide remaining subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.close()


def plot_comparison_distributions(
    original_targets: np.ndarray,
    synthetic_targets: np.ndarray,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot comparison of original vs synthetic class distributions.

    Args:
        original_targets: Original class labels.
        synthetic_targets: Synthetic class labels.
        output_path: Path to save plot.
    """
    unique_orig, counts_orig = np.unique(original_targets, return_counts=True)
    unique_synth, counts_synth = np.unique(synthetic_targets, return_counts=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original distribution
    axes[0].bar(unique_orig, counts_orig, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Original Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Synthetic distribution
    axes[1].bar(unique_synth, counts_synth, color='coral', edgecolor='black')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_title('After Oversampling', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.close()
