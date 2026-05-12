"""Data transformation and augmentation for contrastive learning."""

from typing import Any
import torch
from torch import Tensor
import torchvision.transforms as transforms


def get_cifar10_transform(
    train: bool = True,
    image_size: int = 32,
    horizontal_flip: bool = True,
    crop_padding: int = 4,
) -> transforms.Compose:
    """
    Get CIFAR-10 transformation pipeline for train/test.

    Args:
        train: Whether to use training augmentation.
        image_size: Size of the image.
        horizontal_flip: Whether to apply horizontal flip (train only).
        crop_padding: Padding for random crop (train only).

    Returns:
        Transformation pipeline.
    """
    if train:
        transform_list = [
            transforms.RandomCrop(image_size, padding=crop_padding),
            transforms.RandomHorizontalFlip(p=0.5 if horizontal_flip else 0.0),
        ]
    else:
        transform_list = []

    transform_list.extend([
        transforms.ToTensor(),
    ])

    return transforms.Compose(transform_list)


def get_creditcard_transform(
    train: bool = True,
    dropout_rate: float = 0.1,
    noise_mean: float = 0.0,
    noise_std: float = 0.01,
) -> transforms.Compose:
    """
    Transform for credit card fraud dataset (tabular).

    Args:
        train: Apply augmentation if True.
        dropout_rate: Probability to drop features.
        noise_mean: Mean of Gaussian noise.
        noise_std: Std of Gaussian noise.

    Returns:
        Transformation pipeline.
    """
    class ToTensor:
        def __call__(self, x: Any) -> Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().clone().float()
            return torch.tensor(x, dtype=torch.float32)

    class AddGaussianNoise:
        def __init__(self, mean: float, std: float) -> None:
            self.mean = mean
            self.std = std

        def __call__(self, x: Tensor) -> Tensor:
            noise = torch.randn_like(x) * self.std
            return x + noise + self.mean

    class FeatureDropout:
        def __init__(self, drop_prob: float) -> None:
            self.drop_prob = drop_prob

        def __call__(self, x: Tensor) -> Tensor:
            mask = (torch.rand_like(x) > self.drop_prob).float()
            return x * mask

    transform_list = [ToTensor()]

    if train:
        transform_list.extend([
            FeatureDropout(dropout_rate),
            AddGaussianNoise(noise_mean, noise_std),
        ])

    return transforms.Compose(transform_list)

