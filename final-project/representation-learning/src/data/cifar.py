"""CIFAR-10 data module with controllable long-tail imbalance."""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically.

    Args:
        worker_id: Worker process identifier assigned by DataLoader.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True)
class CIFARDataConfig:
    """Configuration values used by CIFAR-10 dataloaders."""

    root: str
    download: bool
    num_classes: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    imbalance_factor: float
    seed: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    random_crop: bool
    random_horizontal_flip: bool


class ImbalancedCIFAR10(Dataset[Tuple[Tensor, int]]):
    """CIFAR-10 dataset wrapper that applies a long-tail class distribution."""

    def __init__(
        self,
        root: str,
        train: bool,
        imbalance_factor: float,
        num_classes: int,
        transform: transforms.Compose,
        download: bool,
        seed: int,
    ) -> None:
        """Create an imbalanced CIFAR-10 dataset.

        Args:
            root: Dataset root directory.
            train: Whether to load train split.
            imbalance_factor: Ratio between minority and majority classes.
            num_classes: Number of dataset classes.
            transform: Transform pipeline.
            download: Whether torchvision should download dataset if missing.
            seed: Random seed for deterministic class subsampling.
        """
        warning_category: type[Warning] = Warning
        if hasattr(np, "VisibleDeprecationWarning"):
            warning_category = np.VisibleDeprecationWarning
        elif hasattr(np, "exceptions") and hasattr(np.exceptions, "VisibleDeprecationWarning"):
            warning_category = np.exceptions.VisibleDeprecationWarning

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=warning_category,
                module=r"torchvision\.datasets\.cifar",
            )
            self.dataset = CIFAR10(root=root, train=train, transform=transform, download=download)
        targets = np.asarray(self.dataset.targets, dtype=np.int64)

        if train:
            samples_per_class = len(targets) // num_classes
            requested_counts = build_long_tailed_class_counts(
                num_classes=num_classes,
                max_samples_per_class=samples_per_class,
                imbalance_factor=imbalance_factor,
            )
            self.indices = self._sample_indices_per_class(
                targets=targets,
                requested_counts=requested_counts,
                seed=seed,
            )
        else:
            self.indices = np.arange(len(self.dataset), dtype=np.int64)

        selected_targets = targets[self.indices]
        self.class_counts = np.bincount(selected_targets, minlength=num_classes)

    @staticmethod
    def _sample_indices_per_class(
        targets: np.ndarray,
        requested_counts: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Sample deterministic subset indices for each class."""
        rng = np.random.default_rng(seed)
        selected: list[int] = []

        for class_index, class_count in enumerate(requested_counts.tolist()):
            class_indices = np.where(targets == class_index)[0]
            rng.shuffle(class_indices)
            selected.extend(class_indices[:class_count].tolist())

        selected_array = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected_array)
        return selected_array

    def __len__(self) -> int:
        """Return dataset length after imbalance sampling."""
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """Return one transformed sample and class index."""
        sample, label = self.dataset[int(self.indices[index])]
        return sample, int(label)

    def get_class_distribution(self) -> Dict[int, int]:
        """Return per-class sample counts for the current split."""
        return {class_id: int(count) for class_id, count in enumerate(self.class_counts.tolist())}


def build_long_tailed_class_counts(
    num_classes: int,
    max_samples_per_class: int,
    imbalance_factor: float,
) -> np.ndarray:
    """Create exponential long-tail sample counts.

    The class with index 0 is treated as majority, and the last class is minority.

    Args:
        num_classes: Number of classes.
        max_samples_per_class: Maximum samples for the majority class.
        imbalance_factor: Minority/majority ratio in the range (0, 1].

    Returns:
        Integer array containing requested sample counts per class.
    """
    if not (0.0 < imbalance_factor <= 1.0):
        raise ValueError("imbalance_factor must be in the range (0, 1].")

    if num_classes <= 1:
        return np.asarray([max_samples_per_class], dtype=np.int64)

    class_positions = np.linspace(0.0, 1.0, num_classes)
    counts = max_samples_per_class * np.power(imbalance_factor, class_positions)
    counts = np.round(counts).astype(np.int64)
    counts = np.clip(counts, a_min=1, a_max=max_samples_per_class)
    return counts


def create_cifar10_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader[Tuple[Tensor, int]], DataLoader[Tuple[Tensor, int]], Dict[int, int]]:
    """Build train and test dataloaders for imbalanced CIFAR-10.

    Args:
        config: Global experiment configuration dictionary.

    Returns:
        Tuple of train dataloader, test dataloader, and train class distribution.
    """
    parsed = _parse_data_config(config)

    train_transform = _build_train_transform(parsed)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(parsed.mean, parsed.std),
        ]
    )

    train_dataset = ImbalancedCIFAR10(
        root=parsed.root,
        train=True,
        imbalance_factor=parsed.imbalance_factor,
        num_classes=parsed.num_classes,
        transform=train_transform,
        download=parsed.download,
        seed=parsed.seed,
    )
    test_dataset = ImbalancedCIFAR10(
        root=parsed.root,
        train=False,
        imbalance_factor=parsed.imbalance_factor,
        num_classes=parsed.num_classes,
        transform=test_transform,
        download=parsed.download,
        seed=parsed.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(parsed.seed)

    loader_kwargs = {
        "batch_size": parsed.batch_size,
        "num_workers": parsed.num_workers,
        "pin_memory": parsed.pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": parsed.num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, test_loader, train_dataset.get_class_distribution()


def _parse_data_config(config: Dict[str, Any]) -> CIFARDataConfig:
    """Convert nested dictionary config into a typed dataclass."""
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    imbalance_cfg = config["imbalance"]
    augmentation_cfg = dataset_cfg.get("augmentation", {})

    resolved_root = _resolve_cifar_root(str(dataset_cfg["root"]))

    return CIFARDataConfig(
        root=resolved_root,
        download=bool(dataset_cfg.get("download", False)),
        num_classes=int(dataset_cfg.get("num_classes", 10)),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True) and torch.cuda.is_available()),
        imbalance_factor=float(imbalance_cfg["imbalance_factor"]),
        seed=int(config.get("seed", 42)),
        mean=tuple(dataset_cfg.get("mean", [0.4914, 0.4822, 0.4465])),
        std=tuple(dataset_cfg.get("std", [0.2023, 0.1994, 0.2010])),
        random_crop=bool(augmentation_cfg.get("random_crop", True)),
        random_horizontal_flip=bool(augmentation_cfg.get("random_horizontal_flip", True)),
    )


def _resolve_cifar_root(config_root: str) -> str:
    """Resolve CIFAR-10 root for both direct and nested dataset layouts.

    Supports either:
    - <root>/cifar-10-batches-py
    - <root>/cifar10/cifar-10-batches-py
    """
    raw_root = Path(config_root).expanduser()

    if raw_root.is_absolute():
        candidate_roots = [raw_root]
    else:
        project_root = Path(__file__).resolve().parents[2]
        candidate_roots = [
            (Path.cwd() / raw_root).resolve(),
            (project_root / raw_root).resolve(),
        ]

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidate_roots:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            seen.add(candidate_key)
            unique_candidates.append(candidate)

    for root_path in unique_candidates:
        if root_path.name == "cifar-10-batches-py" and root_path.exists():
            return str(root_path.parent)

        direct_layout = root_path / "cifar-10-batches-py"
        nested_layout = root_path / "cifar10" / "cifar-10-batches-py"

        if direct_layout.exists():
            return str(root_path)

        if nested_layout.exists():
            return str(root_path / "cifar10")

    return str(unique_candidates[0] if unique_candidates else raw_root)


def _build_train_transform(parsed: CIFARDataConfig) -> transforms.Compose:
    """Create train-time augmentation and normalization pipeline."""
    augmentation_steps: list[transforms.Transform] = []

    if parsed.random_crop:
        augmentation_steps.append(transforms.RandomCrop(size=32, padding=4))
    if parsed.random_horizontal_flip:
        augmentation_steps.append(transforms.RandomHorizontalFlip())

    augmentation_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(parsed.mean, parsed.std),
        ]
    )

    return transforms.Compose(augmentation_steps)
