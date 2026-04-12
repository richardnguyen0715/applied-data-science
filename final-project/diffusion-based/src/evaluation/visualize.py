"""Visualization utilities."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib import rcParams


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix.
        class_names: Optional class names.
        save_path: Path to save figure.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
) -> None:
    """
    Plot training curves.

    Args:
        history: Training history dictionary.
        save_path: Path to save figure.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    if "loss" in history or "train_loss" in history:
        ax = axes[0]
        if "loss" in history:
            ax.plot(history["epoch"], history["loss"], "b-o", label="Loss", linewidth=2)
        else:
            ax.plot(history["epoch"], history["train_loss"], "b-o", label="Train Loss", linewidth=2)
            if "val_loss" in history:
                ax.plot(history["epoch"], history["val_loss"], "r-o", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Training Loss", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Accuracy plot
    if "train_acc" in history:
        ax = axes[1]
        ax.plot(history["epoch"], history["train_acc"], "b-o", label="Train Acc", linewidth=2)
        if "val_acc" in history:
            ax.plot(history["epoch"], history["val_acc"], "r-o", label="Val Acc", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("Training Accuracy", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_generated_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_samples_per_class: int = 5,
    save_path: Optional[Path] = None,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Plot generated samples.

    Args:
        images: Generated images tensor.
        labels: Corresponding labels.
        num_samples_per_class: Number of samples to show per class.
        save_path: Path to save figure.
        figsize: Figure size.
    """
    # Group by class
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)

    if figsize is None:
        figsize = (num_samples_per_class * 2, num_classes * 1.5)

    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=figsize)
    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, label in enumerate(unique_labels):
        # Get samples for this class
        class_mask = labels == label
        class_images = images[class_mask][:num_samples_per_class]

        for sample_idx, img in enumerate(class_images):
            ax = axes[class_idx, sample_idx]
            # Denormalize if needed (assuming values in [0, 1])
            img_np = img.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = np.clip(img_np / 255.0, 0, 1)
            
            # Convert to display format
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            ax.imshow(img_np, cmap="gray" if img_np.shape[-1] == 1 else None)
            ax.axis("off")

            if sample_idx == 0:
                ax.set_ylabel(f"Class {label.item()}", fontsize=11, fontweight="bold")

    plt.suptitle("Generated Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved generated samples to {save_path}")

    plt.show()


def plot_diffusion_process(
    images: torch.Tensor,
    timesteps: List[int],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot diffusion denoising process.

    Args:
        images: Images at different timesteps.
        timesteps: Corresponding timestep indices.
        save_path: Path to save figure.
    """
    num_steps = len(timesteps)
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))

    for idx, (img, t) in enumerate(zip(images, timesteps)):
        img_np = img.cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = np.clip(img_np / 255.0, 0, 1)

        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        axes[idx].imshow(img_np)
        axes[idx].set_title(f"t={t}", fontsize=10)
        axes[idx].axis("off")

    plt.suptitle("Diffusion Denoising Process", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved diffusion process to {save_path}")

    plt.show()
