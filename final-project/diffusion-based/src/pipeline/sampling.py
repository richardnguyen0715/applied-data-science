"""Sampling utilities for diffusion models."""

from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from src.data.dataset import BalancedDataset
from src.models.diffusion import DiffusionModel


def sample_from_diffusion(
    model: DiffusionModel,
    num_samples_per_class: int = 500,
    image_size: int = 32,
    device: torch.device = None,
    return_numpy: bool = False,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Sample synthetic images from diffusion model.

    Args:
        model: Trained DiffusionModel.
        num_samples_per_class: Number of samples to generate per class.
        image_size: Size of generated images.
        device: Device to sample on.
        return_numpy: Whether to return numpy arrays.

    Returns:
        Tuple of (images, labels).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for class_idx in range(model.num_classes):
            # Generate samples for this class
            samples = torch.randn(
                num_samples_per_class, 3, image_size, image_size, device=device
            )
            labels = torch.full(
                (num_samples_per_class,), class_idx, dtype=torch.long, device=device
            )

            # Reverse diffusion
            pbar = tqdm(
                range(model.num_timesteps - 1, -1, -1),
                desc=f"Sampling class {class_idx}",
                leave=False,
            )
            for t in pbar:
                t_tensor = torch.full(
                    (num_samples_per_class,), t, dtype=torch.long, device=device
                )
                with torch.no_grad():
                    samples = model.denoise_step(samples, t_tensor, labels)

            if return_numpy:
                samples = samples.cpu().numpy()
            else:
                samples = samples.cpu()

            synthetic_images.extend(samples)
            synthetic_labels.extend([class_idx] * num_samples_per_class)

    return synthetic_images, synthetic_labels


def sample_from_diffusion_fast(
    model: DiffusionModel,
    num_samples_per_class: int = 500,
    image_size: int = 32,
    num_steps: int = 100,
    device: torch.device = None,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Fast sampling using fewer diffusion steps (DDIM-like).

    Args:
        model: Trained DiffusionModel.
        num_samples_per_class: Number of samples per class.
        image_size: Size of generated images.
        num_steps: Number of sampling steps.
        device: Device to sample on.

    Returns:
        Tuple of (images, labels).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    synthetic_images = []
    synthetic_labels = []

    # Compute step indices
    step_indices = torch.linspace(
        0, model.num_timesteps - 1, num_steps, dtype=torch.long
    )

    with torch.no_grad():
        for class_idx in range(model.num_classes):
            samples = torch.randn(
                num_samples_per_class, 3, image_size, image_size, device=device
            )
            labels = torch.full(
                (num_samples_per_class,), class_idx, dtype=torch.long, device=device
            )

            # Reverse diffusion with fewer steps
            pbar = tqdm(
                reversed(step_indices),
                desc=f"Fast sampling class {class_idx}",
                total=len(step_indices),
                leave=False,
            )
            for t in pbar:
                t_tensor = torch.full(
                    (num_samples_per_class,), t.item(), dtype=torch.long, device=device
                )
                samples = model.denoise_step(samples, t_tensor, labels)

            synthetic_images.extend(samples.cpu())
            synthetic_labels.extend([class_idx] * num_samples_per_class)

    return synthetic_images, synthetic_labels


def create_balanced_dataset(
    real_dataset,
    synthetic_images: List[torch.Tensor],
    synthetic_labels: List[int],
) -> BalancedDataset:
    """
    Create balanced dataset combining real and synthetic samples.

    Args:
        real_dataset: Original CIFAR10LTDataset.
        synthetic_images: Generated synthetic images.
        synthetic_labels: Labels for synthetic images.

    Returns:
        BalancedDataset instance.
    """
    # Get all real samples
    real_images = []
    real_labels = []

    for idx in range(len(real_dataset)):
        img, label = real_dataset[idx]
        real_images.append(img)
        real_labels.append(label)

    # Combine with synthetic
    all_images = real_images + synthetic_images
    all_labels = real_labels + synthetic_labels

    return BalancedDataset(real_images, real_labels, synthetic_images, synthetic_labels)
