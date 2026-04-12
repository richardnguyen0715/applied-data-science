"""Diffusion model training."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.diffusion import DiffusionModel
from src.training.losses import DiffusionLoss
from src.utils.logger import get_logger, setup_logger


def train_diffusion(
    model: DiffusionModel,
    train_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: torch.device = None,
    checkpoint_dir: Path = None,
    log_dir: Path = None,
    log_every_n_steps: int = 10,
    save_every_n_epochs: int = 10,
    gradient_clip_val: float = 1.0,
    patience: int = 20,
) -> Dict[str, list]:
    """
    Train diffusion model.

    Args:
        model: DiffusionModel instance.
        train_loader: Training DataLoader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        device: Device to train on.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory to save logs.
        log_every_n_steps: Log every n steps.
        save_every_n_epochs: Save checkpoint every n epochs.
        gradient_clip_val: Gradient clipping value.
        patience: Early stopping patience (number of epochs without improvement before stopping).

    Returns:
        Dictionary with training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    if log_dir is not None:
        logger = setup_logger("diffusion", log_dir)
    else:
        logger = get_logger("diffusion")

    # Setup checkpoint directory
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = DiffusionLoss()

    # Training loop
    history = {"loss": [], "epoch": []}
    global_step = 0
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.shape[0]

            # Sample timesteps
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)

            # Sample noise
            noise = torch.randn_like(images)

            # Add noise to images
            x_t = model.q_sample(images, t, noise)

            # Predict noise
            noise_pred = model(x_t, t, labels)

            # Compute loss
            loss = criterion(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()

            # Update EMA
            if model.ema is not None:
                model.ema.update()

            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log
            if (batch_idx + 1) % log_every_n_steps == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Step {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})

        # Epoch statistics
        avg_epoch_loss = epoch_loss / num_batches
        history["loss"].append(avg_epoch_loss)
        history["epoch"].append(epoch + 1)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed | "
            f"Average Loss: {avg_epoch_loss:.4f}"
        )

        # Save best model and check early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1
            patience_counter = 0
            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / "diffusion_best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_epoch_loss,
                    },
                    checkpoint_path,
                )
                logger.info(f"Best model saved to {checkpoint_path} | Loss: {avg_epoch_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement. "
                    f"Best model at epoch {best_epoch} with loss: {best_loss:.4f}"
                )
                break

        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"diffusion_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    logger.info(f"Training completed! Early stopping: {patience_counter >= patience}")
    return history
