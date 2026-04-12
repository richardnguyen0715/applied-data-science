"""Data augmentation and preprocessing transformations."""

from typing import Any, Callable, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image


class RandomCrop:
    """Random crop with padding."""

    def __init__(self, size: int, padding: int = 4) -> None:
        """
        Initialize RandomCrop transform.

        Args:
            size: Target size for cropping.
            padding: Padding before cropping.
        """
        self.size = size
        self.padding = padding

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply random crop.

        Args:
            img: PIL Image to transform.

        Returns:
            Cropped image.
        """
        img = transforms.Pad(self.padding)(img)
        img = transforms.RandomCrop(self.size)(img)
        return img


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize RandomHorizontalFlip.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply random horizontal flip.

        Args:
            img: PIL Image to transform.

        Returns:
            Flipped or original image.
        """
        if torch.rand(1).item() < self.p:
            img = transforms.functional.hflip(img)
        return img


def get_train_transforms(image_size: int = 32) -> Callable:
    """
    Get training data augmentation transforms.

    Args:
        image_size: Size of images.

    Returns:
        Composed transform function.
    """
    return transforms.Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])


def get_test_transforms(image_size: int = 32) -> Callable:
    """
    Get test/validation data transforms (no augmentation).

    Args:
        image_size: Size of images.

    Returns:
        Composed transform function.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])


def get_no_normalizing_transforms(image_size: int = 32) -> Callable:
    """
    Get transforms without normalization (for diffusion model).

    Args:
        image_size: Size of images.

    Returns:
        Composed transform function.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
