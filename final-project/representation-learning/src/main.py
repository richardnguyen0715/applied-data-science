"""Main entry point for CIFAR-10 M2m long-tail training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

from src.data.cifar import create_cifar10_dataloaders
from src.engine.evaluator import Evaluator
from src.engine.trainer import M2MTrainer
from src.models.resnet import build_resnet18
from src.utils.config import load_config
from src.utils.logger import setup_logger


def set_random_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> Optimizer:
    """Build optimizer from training config."""
    train_cfg = config["training"]
    optimizer_name = str(train_cfg.get("optimizer", "sgd")).lower()
    learning_rate = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if optimizer_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> _LRScheduler | None:
    """Build optional learning-rate scheduler from config."""
    train_cfg = config["training"]
    scheduler_name = str(train_cfg.get("scheduler", "none")).lower()

    if scheduler_name == "cosine":
        t_max = int(train_cfg.get("scheduler_t_max", train_cfg["epochs"]))
        return CosineAnnealingLR(optimizer, T_max=t_max)

    if scheduler_name in {"none", ""}:
        return None

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute full training and evaluation pipeline."""
    seed = int(config.get("seed", 42))
    set_random_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = str(config["training"].get("log_dir", "logs"))
    logger = setup_logger(name="m2m_training", log_dir=log_dir, log_file="train.log")
    logger.info("Using device: %s", device)
    logger.info("Random seed: %d", seed)

    train_loader, test_loader, class_distribution = create_cifar10_dataloaders(config)
    logger.info("Imbalanced training distribution: %s", class_distribution)

    num_classes = int(config["dataset"].get("num_classes", 10))
    pretrained = bool(config.get("model", {}).get("pretrained", False))
    model = build_resnet18(num_classes=num_classes, pretrained=pretrained).to(device)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    evaluator = Evaluator(num_classes=num_classes, device=device)
    trainer = M2MTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        evaluator=evaluator,
        config=config,
        device=device,
        logger=logger,
        scheduler=scheduler,
    )

    history: List[Dict[str, float]] = trainer.fit()
    final_metrics = evaluator.evaluate(model, test_loader)

    logger.info("Final accuracy: %.4f", final_metrics["accuracy"])
    logger.info("Final balanced accuracy: %.4f", final_metrics["balanced_accuracy"])
    logger.info("Per-class accuracy: %s", final_metrics["per_class_accuracy"])

    history_path = Path(log_dir) / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)

    return {
        "history": history,
        "final_metrics": final_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train CIFAR-10 with M2m synthesis.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
