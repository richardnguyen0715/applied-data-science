"""Visualize raw vs M2M-synthesized CIFAR-10 images.

Saves a grid with the top row = original source images (major class)
and bottom row = synthesized images targeted to a minority class.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torchvision.utils as vutils

from src.utils.config import load_config
from src.data.cifar import create_cifar10_dataloaders
from src.models.resnet import build_resnet18
from src.m2m.synthesis import M2MSynthesizer


def denormalize(images: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    """Inverse of torchvision Normalize for a batch tensor.

    Args:
        images: Tensor shape (B, C, H, W)
        mean: 3-tuple
        std: 3-tuple

    Returns:
        Denormalized tensor clamped to [0,1].
    """
    device = images.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    denorm = images * std_t + mean_t
    return denorm.clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize M2M synthesis results")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml")
    parser.add_argument("--output", type=str, default="outputs/m2m_visuals")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-sources", type=int, default=8, help="Number of source images to synthesize")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional model checkpoint to load")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of different synthesis examples to create")
    parser.add_argument("--run-prefix", type=str, default="", help="Optional prefix for output filenames")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build data and model
    train_loader, test_loader, train_dist = create_cifar10_dataloaders(config)
    num_classes = int(config["dataset"].get("num_classes", 10))
    model = build_resnet18(num_classes=num_classes, pretrained=False).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    synthesizer = M2MSynthesizer.from_config_dict(config.get("m2m", {}))

    import time
    import numpy as np

    # Access underlying dataset for more diverse sampling
    dataset = train_loader.dataset
    # underlying targets for the imbalanced subset
    underlying_targets = np.asarray(dataset.dataset.targets)[dataset.indices]

    mean = tuple(config["dataset"].get("mean", [0.4914, 0.4822, 0.4465]))
    std = tuple(config["dataset"].get("std", [0.2023, 0.1994, 0.2010]))

    for run_idx in range(int(args.n_runs)):
        # sample a random position to form a source set
        # determine class distribution within a random sample of size batch_size
        rand_idx = np.random.randint(0, len(dataset.indices))
        sample_indices = np.random.choice(len(dataset.indices), size=min(int(args.batch_size), len(dataset.indices)), replace=False)
        sample_targets = underlying_targets[sample_indices]

        counts = torch.tensor(np.bincount(sample_targets, minlength=num_classes), device=device)
        major = int(torch.argmax(counts).item())
        minor = int(torch.argmin(counts).item())

        # find indices within the sampled subset that belong to major class
        major_positions = np.where(sample_targets == major)[0]
        if major_positions.size == 0:
            # fallback: find any major in entire subset
            major_positions = np.where(underlying_targets == major)[0]
            if major_positions.size == 0:
                print("No examples for major class found; skipping run", run_idx)
                continue

        num_sources = min(int(args.num_sources), int(major_positions.size))
        chosen_pos = np.random.choice(major_positions, size=num_sources, replace=False)

        # map chosen positions back to dataset-relative indices and load images
        chosen_dataset_indices = sample_indices[chosen_pos]
        source_images_list = []
        for idx in chosen_dataset_indices:
            img, _ = dataset[int(idx)]
            source_images_list.append(img.unsqueeze(0))
        source_images = torch.cat(source_images_list, dim=0).to(device)

        target_labels = torch.full((num_sources,), fill_value=minor, device=device, dtype=torch.long)

        synthesized = synthesizer.synthesize(model=model, source_images=source_images, target_labels=target_labels)

        src_denorm = denormalize(source_images, mean, std)
        synth_denorm = denormalize(synthesized, mean, std)

        grid = torch.cat([src_denorm, synth_denorm], dim=0)

        run_id = args.run_prefix if args.run_prefix else time.strftime("%Y%m%dT%H%M%S")
        if int(args.n_runs) > 1:
            run_id = f"{run_id}_run{run_idx+1}"

        save_path = out_dir / f"m2m_pairs_{run_id}_major{major}_minor{minor}.png"
        vutils.save_image(grid, str(save_path), nrow=num_sources, pad_value=0.02)

        print("Saved M2M visualization to:", save_path)


if __name__ == "__main__":
    main()
