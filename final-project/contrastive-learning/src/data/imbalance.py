"""Imbalance analysis utilities."""

from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from src.utils.logger import get_logger


def analyze_class_distribution(dataset: Dataset) -> Dict[int, int]:
    """
    Analyze class distribution in dataset.

    Args:
        dataset: PyTorch dataset.

    Returns:
        Dictionary with class -> count mapping.
    """
    logger = get_logger("imbalance")

    class_counts = {}
    if hasattr(dataset, "labels"):
        if isinstance(dataset.labels, np.ndarray):
            labels = dataset.labels
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    for label in labels:
        class_counts[int(label)] = class_counts.get(int(label), 0) + 1

    return class_counts


def print_class_distribution(class_counts: Dict[int, int]) -> None:
    """
    Print class distribution.

    Args:
        class_counts: Dictionary with class -> count mapping.
    """
    logger = get_logger("imbalance")

    logger.info("Class distribution:")
    total = sum(class_counts.values())
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = 100 * count / total
        logger.info(f"  Class {class_id}: {count:6d} ({percentage:5.2f}%)")

    # Calculate imbalance ratio
    if len(class_counts) > 0:
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}x")


def get_class_weights(class_counts: Dict[int, int], num_classes: int) -> np.ndarray:
    """
    Get class weights for handling imbalanced data.

    Args:
        class_counts: Dictionary with class -> count mapping.
        num_classes: Total number of classes.

    Returns:
        Array of class weights.
    """
    weights = np.zeros(num_classes)
    total = sum(class_counts.values())

    for class_id in range(num_classes):
        if class_id in class_counts:
            weights[class_id] = total / (num_classes * class_counts[class_id])
        else:
            weights[class_id] = 1.0

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return weights


def create_balanced_sampler(
    dataset: Dataset,
    num_classes: int,
    seed: int = 42,
):
    """
    Create a balanced sampler for handling imbalanced data.

    Args:
        dataset: PyTorch dataset.
        num_classes: Number of classes.
        seed: Random seed.

    Returns:
        BalancedBatchSampler instance.
    """
    from torch.utils.data import Sampler

    class BalancedBatchSampler(Sampler):
        """Sampler that returns balanced batches."""

        def __init__(self, dataset, batch_size=64, num_classes=10, seed=42):
            """Initialize balanced batch sampler."""
            self.dataset = dataset
            self.batch_size = batch_size
            self.num_classes = num_classes
            self.seed = seed

            # Get class indices
            self.class_indices = {}
            for i in range(len(dataset)):
                label = dataset[i][1]
                if label not in self.class_indices:
                    self.class_indices[label] = []
                self.class_indices[label].append(i)

            # Shuffle indices
            rng = np.random.RandomState(seed)
            for label in self.class_indices:
                rng.shuffle(self.class_indices[label])

        def __iter__(self):
            """Iterate over balanced batches."""
            batch = []
            class_id = 0
            while True:
                for _ in range(self.batch_size // self.num_classes):
                    if not self.class_indices[class_id]:
                        return
                    batch.append(self.class_indices[class_id].pop())

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

                class_id = (class_id + 1) % self.num_classes

        def __len__(self):
            """Return number of batches."""
            return len(self.dataset) // self.batch_size

    return BalancedBatchSampler(
        dataset, batch_size=32, num_classes=num_classes, seed=seed
    )
