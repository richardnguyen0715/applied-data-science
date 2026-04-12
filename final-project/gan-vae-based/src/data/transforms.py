"""Data transformations and augmentations."""

from typing import Callable

import torch
import torchvision.transforms as transforms


def get_train_transforms(image_size: int = 32) -> Callable:
    """
    Get training data transformations.

    Args:
        image_size: Size of images.

    Returns:
        Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


def get_test_transforms(image_size: int = 32) -> Callable:
    """
    Get test/validation data transformations.

    Args:
        image_size: Size of images.

    Returns:
        Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


def get_generative_transforms(image_size: int = 32) -> Callable:
    """
    Get transformations for generative models (simpler than train).

    Args:
        image_size: Size of images.

    Returns:
        Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple = (0.2023, 0.1994, 0.2010),
) -> torch.Tensor:
    """
    Denormalize tensor back to original scale.

    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W).
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Denormalized tensor.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    mean_tensor = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)

    denormalized = tensor * std_tensor + mean_tensor

    if squeeze:
        denormalized = denormalized.squeeze(0)

    return denormalized.clamp(0, 1)
