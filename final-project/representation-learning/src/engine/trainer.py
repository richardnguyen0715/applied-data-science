"""Training engine with warm-up and on-the-fly M2m augmentation."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.engine.evaluator import Evaluator
from src.m2m.synthesis import M2MSynthesizer


class M2MTrainer:
    """Train a classifier with dynamic Major-to-Minor synthesis."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        evaluator: Evaluator,
        config: Dict[str, Any],
        device: torch.device,
        logger: Any,
        scheduler: _LRScheduler | None = None,
    ) -> None:
        """Initialize trainer state.

        Args:
            model: Classifier model.
            optimizer: Training optimizer.
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            evaluator: Evaluator instance.
            config: Full experiment config.
            device: Compute device.
            logger: Logger object.
            scheduler: Optional learning-rate scheduler.
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.config = config
        self.device = device
        self.logger = logger
        self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss()
        self.epochs = int(config["training"]["epochs"])
        self.warmup_epochs = int(config["warmup_epochs"])
        self.max_synth_per_batch = int(config["training"].get("max_synth_per_batch", 16))
        self.log_interval = int(config["training"].get("log_interval", 50))
        self.num_classes = int(config["dataset"].get("num_classes", 10))

        self.use_amp = bool(config["training"].get("amp", False)) and device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.synthesizer = M2MSynthesizer.from_config_dict(config["m2m"])

        self.checkpoint_dir = Path(config["training"].get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one epoch of supervised training.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            Dictionary with training loss, accuracy, and synthesized sample count.
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        total_synthesized = 0

        for step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            augmented_images, augmented_labels, synthesized_count = self._build_augmented_batch(
                images=images,
                labels=labels,
                epoch=epoch,
            )

            self.optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else nullcontext()
            )

            with autocast_context:
                logits = self.model(augmented_images)
                loss = self.criterion(logits, augmented_labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            predictions = torch.argmax(logits, dim=1)

            batch_size = augmented_labels.size(0)
            total_examples += batch_size
            total_loss += float(loss.item() * batch_size)
            total_correct += int((predictions == augmented_labels).sum().item())
            total_synthesized += synthesized_count

            if (step + 1) % self.log_interval == 0:
                self.logger.info(
                    "Epoch %d Step %d/%d | loss=%.4f | synthesized=%d",
                    epoch + 1,
                    step + 1,
                    len(self.train_loader),
                    float(loss.item()),
                    synthesized_count,
                )

        average_loss = total_loss / max(total_examples, 1)
        average_accuracy = total_correct / max(total_examples, 1)

        return {
            "loss": float(average_loss),
            "accuracy": float(average_accuracy),
            "num_synthesized": float(total_synthesized),
        }

    def fit(self) -> List[Dict[str, float]]:
        """Run full training loop and return epoch history."""
        history: List[Dict[str, float]] = []
        best_balanced_accuracy = -1.0

        for epoch in range(self.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluator.evaluate(self.model, self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_log = {
                "epoch": float(epoch + 1),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "num_synthesized": train_metrics["num_synthesized"],
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            }
            history.append(epoch_log)

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | train_acc=%.4f | "
                "val_loss=%.4f | val_acc=%.4f | val_bal_acc=%.4f | synthesized=%.0f",
                epoch + 1,
                self.epochs,
                epoch_log["train_loss"],
                epoch_log["train_accuracy"],
                epoch_log["val_loss"],
                epoch_log["val_accuracy"],
                epoch_log["val_balanced_accuracy"],
                epoch_log["num_synthesized"],
            )

            self._save_checkpoint(
                file_name="last.pt",
                epoch=epoch,
                metrics=epoch_log,
            )

            if epoch_log["val_balanced_accuracy"] > best_balanced_accuracy:
                best_balanced_accuracy = epoch_log["val_balanced_accuracy"]
                self._save_checkpoint(
                    file_name="best.pt",
                    epoch=epoch,
                    metrics=epoch_log,
                )

        return history

    def _build_augmented_batch(self, images: Tensor, labels: Tensor, epoch: int) -> Tuple[Tensor, Tensor, int]:
        """Generate near-balanced augmented batches with M2m after warm-up.

        Args:
            images: Input image batch.
            labels: Input labels.
            epoch: Zero-based epoch index.

        Returns:
            Augmented images, augmented labels, and synthesized sample count.
        """
        if epoch < self.warmup_epochs or self.max_synth_per_batch <= 0:
            return images, labels, 0

        class_counts = torch.bincount(labels, minlength=self.num_classes)
        present_classes = torch.where(class_counts > 0)[0]

        if present_classes.numel() < 2:
            return images, labels, 0

        target_count = int(class_counts[present_classes].max().item())
        effective_counts = class_counts.clone()
        synthesis_budget = self.max_synth_per_batch

        synthesized_images: List[Tensor] = []
        synthesized_labels: List[Tensor] = []

        while synthesis_budget > 0:
            major_class = int(torch.argmax(effective_counts).item())
            minor_candidates = [
                int(class_index)
                for class_index in present_classes.tolist()
                if class_index != major_class and int(effective_counts[class_index].item()) < target_count
            ]

            if not minor_candidates:
                break

            minor_class = min(minor_candidates, key=lambda idx: int(effective_counts[idx].item()))
            deficit = target_count - int(effective_counts[minor_class].item())
            if deficit <= 0:
                break

            source_indices = torch.where(labels == major_class)[0]
            if source_indices.numel() == 0:
                break

            synth_count = min(deficit, synthesis_budget, int(source_indices.numel()))
            if synth_count <= 0:
                break

            selected_positions = torch.randperm(source_indices.numel(), device=labels.device)[:synth_count]
            selected_indices = source_indices[selected_positions]

            source_images = images[selected_indices].detach()
            target_labels = torch.full(
                (synth_count,),
                fill_value=minor_class,
                device=labels.device,
                dtype=torch.long,
            )

            generated_images = self.synthesizer.synthesize(
                model=self.model,
                source_images=source_images,
                target_labels=target_labels,
            )

            synthesized_images.append(generated_images)
            synthesized_labels.append(target_labels)

            effective_counts[minor_class] += synth_count
            synthesis_budget -= synth_count

        if not synthesized_images:
            return images, labels, 0

        synthetic_batch = torch.cat(synthesized_images, dim=0)
        synthetic_targets = torch.cat(synthesized_labels, dim=0)

        augmented_images = torch.cat([images, synthetic_batch], dim=0)
        augmented_labels = torch.cat([labels, synthetic_targets], dim=0)

        permutation = torch.randperm(augmented_images.size(0), device=augmented_images.device)
        augmented_images = augmented_images[permutation]
        augmented_labels = augmented_labels[permutation]

        return augmented_images, augmented_labels, int(synthetic_batch.size(0))

    def _save_checkpoint(self, file_name: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Persist model checkpoint to disk."""
        checkpoint_path = self.checkpoint_dir / file_name
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(payload, checkpoint_path)
