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
        contrastive: bool = True,
    ) -> None:
        """
        Initialize contrastive dataset.

        Args:
            split: Data split ('train', 'val', or 'test').
            transform: Optional transforms to apply.
            contrastive: If True, create 2 augmented views for contrastive learning.
                        If False, create only 1 view for evaluation.
        """
        self.split = split
        self.transform = transform
        self.contrastive = contrastive

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
        contrastive: bool = False,
        dataset_name: str = "tomas-gajarsky/cifar10-lt",
        dataset_config: str = "r-100",
    ) -> None:
        """
        Initialize CIFAR-10-LT dataset.

        Args:
            split: Data split ('train', 'val' or 'test').
            transform: Optional transforms to apply.
            contrastive: If True, create 2 augmented views for contrastive learning.
                         If False, create only 1 view for evaluation.
            dataset_name: HuggingFace dataset name.
            dataset_config: Dataset configuration (e.g., 'r-100').
        """
        super().__init__(split=split, transform=transform, contrastive=contrastive)

        # Load from HuggingFace
        ds = load_dataset(dataset_name, dataset_config, split=split)
        if split == 'val':
            # HuggingFace doesn't have a separate val split, so we create it from train
            ds = load_dataset(dataset_name, dataset_config, split="train")
            ds = ds.train_test_split(test_size=0.1, stratify_by_column="label", seed=1)["test"]

        self.data = ds
        self.labels = np.array(self.data["label"])

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample from the dataset.
        
        Returns:
            If contrastive=True and transform is provided: ((x_i, x_j), label) for contrastive learning
            If contrastive=False and transform is provided: (x, label) for evaluation
            If transform is None: (img, label) for evaluation
        """
        sample = self.data[idx]
        img = sample["img"]  # PIL Image
        label = sample["label"]

        if self.transform is not None:
            if self.contrastive:
                # Create 2 augmented views for contrastive learning
                x_i = self.transform(img)
                x_j = self.transform(img)
                return (x_i, x_j), label
            else:
                # Create only 1 augmented view for evaluation
                x = self.transform(img)
                return x, label
        else:
            # Return raw image for evaluation
            return img, label


class CreditCardFraudDataset(ContrastiveDataset):
    """Credit Card Fraud Detection dataset for contrastive learning."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Compose] = None,
        contrastive: bool = False,
        data_path: Path = None,
        normalize: bool = True,
    ) -> None:
        """
        Initialize Credit Card Fraud dataset.

        Args:
            split: Data split ('train' or 'test').
            transform: Optional transforms to apply.
            contrastive: If True, create 2 augmented views for contrastive learning.
                         If False, create only 1 view for evaluation.
            data_path: Path to CSV file.
            normalize: Whether to normalize features.
        """
        super().__init__(split=split, transform=transform, contrastive=contrastive)

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
            X, y, test_size=0.2, random_state=1, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1, random_state=1, stratify=y_temp
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
            If contrastive=True and transform is provided: ((x_i, x_j), label) for contrastive learning
            If contrastive=False and transform is provided: (x, label) for evaluation
            If transform is None: (x, label) for evaluation
        """
        x = self.data[idx]  # (30,) features
        label = self.labels[idx].item()

        if self.transform is not None:
            if self.contrastive:
                # Create 2 augmented views for contrastive learning
                x_i = self.transform(x)
                x_j = self.transform(x)
                return (x_i, x_j), label
            else:
                # Create only 1 augmented view for evaluation
                x_aug = self.transform(x)
                return x_aug, label
        else:
            # Return raw features for evaluation
            return x, label