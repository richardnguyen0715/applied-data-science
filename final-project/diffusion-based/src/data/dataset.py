"""CIFAR-10-LT dataset loading and processing."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from src.utils.logger import setup_logger


class CIFAR10LTDataset(Dataset):
    """PyTorch Dataset for CIFAR-10-LT."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Compose] = None,
        dataset_name: str = "tomas-gajarsky/cifar10-lt",
        dataset_config: str = "r-20",
    ) -> None:
        """
        Initialize CIFAR-10-LT dataset.

        Args:
            split: Data split ('train' or 'test').
            transform: Optional transforms to apply.
            dataset_name: HuggingFace dataset name.
            dataset_config: Dataset configuration (e.g., 'r-20').
        """
        self.split = split
        self.transform = transform

        # Load from HuggingFace
        ds = load_dataset(dataset_name, dataset_config, split=split)
        self.data = ds

        # Cache labels for quick access
        self.labels = self.data["label"]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, label).
        """
        sample = self.data[idx]
        img = sample["img"]
        label = sample["label"]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class BalancedDataset(Dataset):
    """Balanced dataset combining real and synthetic samples."""

    def __init__(
        self,
        real_images: List[torch.Tensor],
        real_labels: List[int],
        synthetic_images: List[torch.Tensor],
        synthetic_labels: List[int],
    ) -> None:
        """
        Initialize balanced dataset.

        Args:
            real_images: List of real image tensors.
            real_labels: List of real labels.
            synthetic_images: List of synthetic image tensors.
            synthetic_labels: List of synthetic labels.
        """
        self.images = real_images + synthetic_images
        self.labels = real_labels + synthetic_labels

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, label).
        """
        return self.images[idx], self.labels[idx]


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for a dataset.

    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory.
        shuffle: Whether to shuffle data.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )


def load_cifar10lt(
    split: str = "train",
    transform: Optional[Compose] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    dataset_name: str = "tomas-gajarsky/cifar10-lt",
    dataset_config: str = "r-20",
) -> Tuple[CIFAR10LTDataset, DataLoader]:
    """
    Load CIFAR-10-LT dataset and create DataLoader.

    Args:
        split: Data split ('train' or 'test').
        transform: Optional transforms.
        batch_size: Batch size.
        num_workers: Number of workers.
        shuffle: Whether to shuffle.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration.

    Returns:
        Tuple of (Dataset, DataLoader).
    """
    dataset = CIFAR10LTDataset(
        split=split,
        transform=transform,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )
    dataloader = create_data_loaders(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return dataset, dataloader


def get_class_distribution(dataset: CIFAR10LTDataset) -> Dict[int, int]:
    """
    Get class distribution of a dataset.

    Args:
        dataset: CIFAR10LTDataset instance.

    Returns:
        Dictionary mapping class to count.
    """
    from collections import Counter
    labels = [dataset.labels[i] for i in range(len(dataset))]
    return dict(sorted(Counter(labels).items()))
