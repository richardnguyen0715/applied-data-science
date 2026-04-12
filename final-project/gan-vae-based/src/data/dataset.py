"""Dataset loading and PyTorch Dataset wrapper."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset

from src.data.transforms import get_test_transforms, get_train_transforms


class CIFAR10LTDataset(Dataset):
    """PyTorch Dataset wrapper for CIFAR-10-LT from HuggingFace."""

    def __init__(
        self,
        hf_dataset: torch.utils.data.Dataset,
        transform=None,
    ):
        """
        Initialize dataset.

        Args:
            hf_dataset: HuggingFace dataset.
            transform: Optional torchvision transforms.
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.raw_dataset = hf_dataset  # Store reference to raw dataset for label extraction

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.

        Args:
            idx: Index.

        Returns:
            Tuple of (image, label).
        """
        sample = self.dataset[idx]
        
        # Handle both dict and tuple formats from HuggingFace
        if isinstance(sample, dict):
            # HuggingFace CIFAR-10-LT uses 'img' not 'image'
            image = sample.get('img') or sample.get('image')
            label = sample['label']
        else:
            # If it's already processed, assume it's (image, label) tuple
            image, label = sample[0], sample[1]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def get_targets(self) -> list:
        """
        Get all target labels without applying transforms.

        Returns:
            List of target labels.
        """
        targets = []
        for sample in self.raw_dataset:
            if isinstance(sample, dict):
                targets.append(sample['label'])
            else:
                targets.append(sample[1])
        return targets
    
    def get_images_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all images (as transformed tensors) and targets.
        
        Returns:
            Tuple of (images_array, targets_array) where images are transformed.
        """
        images = []
        targets = []
        for i in range(len(self)):
            img, label = self[i]
            # Convert tensor to numpy if needed
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            images.append(img)
            targets.append(label)
        return np.array(images), np.array(targets)


def load_cifar10lt(
    dataset_name: str = "tomas-gajarsky/cifar10-lt",
    config_name: str = "r-20",
    cache_dir: Optional[Path] = None,
) -> DatasetDict:
    """
    Load CIFAR-10-LT dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier.
        config_name: Configuration name (e.g., 'r-20').
        cache_dir: Cache directory for downloaded data.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    kwargs = {}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)

    ds = load_dataset(dataset_name, config_name, **kwargs)

    return ds


def get_train_val_datasets(
    dataset_name: str = "tomas-gajarsky/cifar10-lt",
    config_name: str = "r-20",
    val_split: float = 0.1,
    image_size: int = 32,
    cache_dir: Optional[Path] = None,
) -> Tuple[CIFAR10LTDataset, CIFAR10LTDataset, CIFAR10LTDataset]:
    """
    Get train, validation, and test datasets.

    Args:
        dataset_name: HuggingFace dataset identifier.
        config_name: Configuration name.
        val_split: Fraction of training data to use for validation.
        image_size: Size of images.
        cache_dir: Cache directory.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    ds = load_cifar10lt(dataset_name, config_name, cache_dir)

    # Split training data into train and val
    train_split = ds['train'].train_test_split(
        test_size=val_split,
        seed=42,
        stratify_by_column='label',
    )

    train_transform = get_train_transforms(image_size)
    test_transform = get_test_transforms(image_size)

    train_dataset = CIFAR10LTDataset(
        train_split['train'],
        transform=train_transform,
    )
    val_dataset = CIFAR10LTDataset(
        train_split['test'],
        transform=test_transform,
    )
    test_dataset = CIFAR10LTDataset(
        ds['test'],
        transform=test_transform,
    )

    return train_dataset, val_dataset, test_dataset
