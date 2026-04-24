"""Contrastive encoder training."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.encoder import ContrastiveEncoder
from src.training.losses import SupConLoss
from src.utils.logger import get_logger, setup_logger


def train_contrastive_encoder(
    model: ContrastiveEncoder,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 200,
    learning_rate: float = 0.5,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    temperature: float = 0.07,
    warmup_epochs: int = 0,
    cosine_annealing: bool = True,
    device: torch.device = None,
    checkpoint_dir: Path = None,
    log_dir: Path = None,
    log_every_n_steps: int = 10,
    save_every_n_epochs: int = 10,
    gradient_clip_val: float = 0.0,
    patience: int = 20,
) -> Dict[str, list]:
    """
    Train contrastive encoder.

    Args:
        model: Contrastive encoder model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (optional).
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularization.
        momentum: Momentum for SGD optimizer.
        temperature: Temperature for contrastive loss.
        warmup_epochs: Number of warmup epochs.
        cosine_annealing: Whether to use cosine annealing.
        device: Device to train on.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory to save logs.
        log_every_n_steps: Log every n steps.
        save_every_n_epochs: Save checkpoint every n epochs.
        gradient_clip_val: Gradient clipping value.

    Returns:
        Dictionary with training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log_dir is not None:
        logger = setup_logger("contrastive_training", log_dir)
    else:
        logger = get_logger("contrastive_training")

    model = model.to(device)

    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Setup loss
    criterion = SupConLoss(temperature=temperature)

    # Setup scheduler
    scheduler = None
    if cosine_annealing:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, ((x_i, x_j), labels) in enumerate(pbar):
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            # Compute loss
            loss = criterion(z_i, z_j, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % log_every_n_steps == 0:
                avg_loss = train_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        avg_train_loss = train_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # Learning rate step
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"LR: {current_lr:.6f}"
        )

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for (x_i, x_j), labels in val_loader:
                    x_i = x_i.to(device)
                    x_j = x_j.to(device)
                    labels = labels.to(device)

                    _, z_i = model(x_i)
                    _, z_j = model(x_j)

                    loss = criterion(z_i, z_j, labels)

                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            history["val_loss"].append(avg_val_loss)

            logger.info(f"Val Loss: {avg_val_loss:.4f}")

            # Save checkpoint if val loss improved
            if checkpoint_dir is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve_epochs = 0

                    checkpoint_path = checkpoint_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_val_loss,
                    }, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
                else:
                    no_improve_epochs += 1

                # Early stopping
                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Save checkpoint periodically
        if checkpoint_dir is not None and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
            }, checkpoint_path)

    return history
