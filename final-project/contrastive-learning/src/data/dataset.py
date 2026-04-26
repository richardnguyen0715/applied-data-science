"""Dataset loading and processing for contrastive learning."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from src.utils.logger import get_logger


class ContrastiveDataset(Dataset):
    """Base class for contrastive learning datasets."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Compose] = None,
    ) -> None:
        """
        Initialize contrastive dataset.

        Args:
            split: Data split ('train', 'val', or 'test').
            transform: Optional transforms to apply.
        """
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple:
        """Get a sample from the dataset."""
        raise NotImplementedError


class CIFAR10LTContrastiveDataset(ContrastiveDataset):
    """CIFAR-10-LT dataset for contrastive learning."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Compose] = None,
        dataset_name: str = "tomas-gajarsky/cifar10-lt",
        dataset_config: str = "r-100",
    ) -> None:
        """
        Initialize CIFAR-10-LT dataset.

        Args:
            split: Data split ('train' or 'test').
            transform: Optional transforms to apply.
            dataset_name: HuggingFace dataset name.
            dataset_config: Dataset configuration (e.g., 'r-100').
        """
        super().__init__(split=split, transform=transform)

        # Load from HuggingFace
        ds = load_dataset(dataset_name, dataset_config, split=split)
        self.data = ds
        self.labels = np.array(self.data["label"])

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample from the dataset.
        
        Returns:
            If transform is provided: ((x_i, x_j), label) for contrastive learning
            If transform is None: (img, label) for evaluation
        """
        sample = self.data[idx]
        img = sample["img"]  # PIL Image
        label = sample["label"]

        if self.transform is not None:
            # Apply transform twice to get two different augmented views
            x_i = self.transform(img)
            x_j = self.transform(img)
            return (x_i, x_j), label
        else:
            # Return raw image for evaluation
            return img, label


class CreditCardFraudDataset(ContrastiveDataset):
    """Credit Card Fraud Detection dataset for contrastive learning."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Compose] = None,
        data_path: Path = None,
        normalize: bool = True,
    ) -> None:
        """
        Initialize Credit Card Fraud dataset.

        Args:
            split: Data split ('train', 'val', or 'test').
            transform: Optional transforms to apply.
            data_path: Path to CSV file.
            normalize: Whether to normalize features.
        """
        super().__init__(split=split, transform=transform)

        logger = get_logger("dataset")

        if data_path is None:
            data_path = Path("data/creditcard.csv")

        if not data_path.exists():
            logger.warning(f"Dataset not found at {data_path}. Please download from Kaggle.")
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                "Please download creditcard.csv from Kaggle."
            )

        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)

        # Prepare features and labels
        X = df.drop("Class", axis=1).values
        y = df["Class"].values

        # Normalize features
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        X = X.astype(np.float32)

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
        )

        if split == "train":
            self.data = torch.from_numpy(X_train).float()
            self.labels = torch.from_numpy(y_train).long()
        elif split == "val":
            self.data = torch.from_numpy(X_val).float()
            self.labels = torch.from_numpy(y_val).long()
        else:  # test
            self.data = torch.from_numpy(X_test).float()
            self.labels = torch.from_numpy(y_test).long()

        logger.info(f"{split} split: {len(self.data)} samples")
        logger.info(f"Class distribution: {np.bincount(self.labels.numpy())}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample from the dataset.
        
        Returns:
            If transform is provided: ((x_i, x_j), label) for contrastive learning
            If transform is None: (x, label) for evaluation
        """
        x = self.data[idx]  # (30,) features
        label = self.labels[idx].item()

        if self.transform is not None:
            # For tabular data, apply IdentityTransform (no augmentation) to get two identical views
            x_i = self.transform(x)
            x_j = self.transform(x)
            return (x_i, x_j), label
        else:
            # Return raw features for evaluation
            return x, label


def create_contrastive_dataset(
    dataset_name: str = "cifar10-lt",
    split: str = "train",
    transform: Optional[Compose] = None,
    data_path: Optional[Path] = None,
    **kwargs,
) -> ContrastiveDataset:
    """
    Create contrastive dataset.

    Args:
        dataset_name: Dataset name ('cifar10-lt' or 'credit-card-fraud').
        split: Data split.
        transform: Transforms to apply.
        data_path: Path to data file (for credit card fraud).
        **kwargs: Additional arguments.

    Returns:
        Dataset instance.
    """
    if dataset_name == "cifar10-lt":
        return CIFAR10LTContrastiveDataset(
            split=split,
            transform=transform,
            dataset_config=kwargs.get("dataset_config", "r-100"),
        )
    elif dataset_name == "credit-card-fraud":
        return CreditCardFraudDataset(
            split=split,
            transform=transform,
            data_path=data_path,
            normalize=kwargs.get("normalize", True),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create data loaders.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size.
        num_workers: Number of workers.
        shuffle_train: Whether to shuffle training data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
