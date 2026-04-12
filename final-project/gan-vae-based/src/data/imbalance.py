"""Utilities for analyzing and handling imbalanced datasets."""

from typing import List, Tuple

import numpy as np
import torch


def get_class_distribution(
    targets: torch.Tensor | List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get class distribution statistics.

    Args:
        targets: Target labels.

    Returns:
        Tuple of (class_indices, counts) sorted by count descending.
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    targets = np.array(targets)

    unique, counts = np.unique(targets, return_counts=True)
    # Sort by count descending
    sorted_indices = np.argsort(-counts)

    return unique[sorted_indices], counts[sorted_indices]


def get_imbalance_ratio(
    targets: torch.Tensor | List[int],
) -> float:
    """
    Calculate imbalance ratio (max_class_count / min_class_count).

    Args:
        targets: Target labels.

    Returns:
        Imbalance ratio.
    """
    _, counts = get_class_distribution(targets)
    return float(counts.max() / counts.min())


def identify_minority_classes(
    targets: torch.Tensor | List[int],
    threshold: float = 0.5,
) -> List[int]:
    """
    Identify minority classes based on threshold.

    Args:
        targets: Target labels.
        threshold: Percentile threshold for minority (0-1).

    Returns:
        List of minority class indices.
    """
    unique, counts = get_class_distribution(targets)
    percentile = np.percentile(counts, threshold * 100)
    minority_classes = unique[counts <= percentile].tolist()

    return sorted(minority_classes)


def get_samples_needed(
    targets: torch.Tensor | List[int],
    target_ratio: float = 1.0,
) -> dict[int, int]:
    """
    Calculate number of samples needed per minority class to achieve target ratio.

    Args:
        targets: Target labels.
        target_ratio: Target imbalance ratio (1.0 = balanced, >1.0 = imbalanced).

    Returns:
        Dict mapping class index to number of samples to generate.
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    targets = np.array(targets)

    unique, counts = np.unique(targets, return_counts=True)
    max_count = counts.max()
    target_count = int(max_count / target_ratio)

    samples_needed = {}
    for class_idx, count in zip(unique, counts):
        if count < target_count:
            samples_needed[int(class_idx)] = target_count - count

    return samples_needed


def create_balanced_subset(
    targets: torch.Tensor | List[int],
    indices: np.ndarray | List[int] | None = None,
    strategy: str = "oversample",
) -> np.ndarray:
    """
    Create balanced dataset indices using specified strategy.

    Args:
        targets: Target labels.
        indices: Available indices (if None, use all).
        strategy: 'oversample' or 'undersample'.

    Returns:
        Array of balanced indices.
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    targets = np.array(targets)

    if indices is None:
        indices = np.arange(len(targets))
    else:
        indices = np.array(indices)

    unique, counts = np.unique(targets, return_counts=True)

    if strategy == "oversample":
        max_count = counts.max()
        balanced_indices = []
        for class_idx in unique:
            class_mask = targets == class_idx
            class_indices = indices[class_mask]
            oversampled = np.random.choice(
                class_indices,
                size=max_count,
                replace=True,
            )
            balanced_indices.extend(oversampled)
    elif strategy == "undersample":
        min_count = counts.min()
        balanced_indices = []
        for class_idx in unique:
            class_mask = targets == class_idx
            class_indices = indices[class_mask]
            undersampled = np.random.choice(
                class_indices,
                size=min_count,
                replace=False,
            )
            balanced_indices.extend(undersampled)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return np.array(balanced_indices)
