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


def get_creditcard_transform() -> transforms.Compose:
    """
    Get transformation for credit card fraud dataset (tabular data).
    
    Returns:
        A callable transform that applies feature dropout and Gaussian noise.
    """
    class ToTensor:
        def __call__(self, x: Any) -> Tensor:
            return torch.tensor(x, dtype=torch.float32)


    class AddGaussianNoise:
        def __init__(self, mean: float = 0.0, std: float = 0.01) -> None:
            self.mean = mean
            self.std = std

        def __call__(self, x: Tensor) -> Tensor:
            noise: Tensor = torch.randn_like(x) * self.std
            return x + noise + self.mean


    class FeatureDropout:
        def __init__(self, drop_prob: float = 0.1) -> None:
            self.drop_prob = drop_prob

        def __call__(self, x: Tensor) -> Tensor:
            mask: Tensor = (torch.rand_like(x) > self.drop_prob).float()
            return x * mask
    
    return transforms.Compose([
        ToTensor(),
        FeatureDropout(0.1),
        AddGaussianNoise(0.0, 0.01)
    ])

