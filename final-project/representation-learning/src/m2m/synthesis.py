"""Major-to-Minor (M2m) synthesis for long-tailed learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class M2MConfig:
    """Configuration for the M2m synthesis process."""

    steps: int
    step_size: float
    lambda_identity: float


class M2MSynthesizer:
    """Generate synthetic minority samples from majority examples."""

    def __init__(self, config: M2MConfig) -> None:
        """Initialize synthesizer.

        Args:
            config: Synthesis hyperparameters.
        """
        self.config = config

    @classmethod
    def from_config_dict(cls, config: Dict[str, float]) -> "M2MSynthesizer":
        """Build synthesizer from dictionary configuration."""
        return cls(
            M2MConfig(
                steps=int(config["steps"]),
                step_size=float(config["lr"]),
                lambda_identity=float(config["lambda"]),
            )
        )

    def synthesize(self, model: nn.Module, source_images: Tensor, target_labels: Tensor) -> Tensor:
        """Synthesize samples using configured M2m settings."""
        return synthesize_m2m(
            model=model,
            source_images=source_images,
            target_labels=target_labels,
            steps=self.config.steps,
            step_size=self.config.step_size,
            lambda_identity=self.config.lambda_identity,
        )


def synthesize_m2m(
    model: nn.Module,
    source_images: Tensor,
    target_labels: Tensor,
    steps: int,
    step_size: float,
    lambda_identity: float,
    clamp_range: Tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """Perform M2m synthesis with target CE + identity MSE and FGSM-like updates.

    Args:
        model: Classifier used as guidance model.
        source_images: Majority-class images to be transformed.
        target_labels: Target minority labels.
        steps: Number of synthesis refinement steps.
        step_size: Sign-gradient update magnitude per step.
        lambda_identity: Identity preservation weight.
        clamp_range: Valid pixel range.

    Returns:
        Synthesized image tensor with the same shape as source images.
    """
    if source_images.ndim != 4:
        raise ValueError("source_images must have shape (N, C, H, W).")
    if target_labels.ndim != 1:
        raise ValueError("target_labels must have shape (N,).")
    if source_images.size(0) != target_labels.size(0):
        raise ValueError("source_images and target_labels must have the same batch size.")

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    synthesized = source_images.detach().clone()
    identity_anchor = source_images.detach().clone()
    was_training = model.training

    try:
        model.eval()
        for _ in range(steps):
            synthesized.requires_grad_(True)

            logits = model(synthesized)
            target_term = ce_loss(logits, target_labels)
            identity_term = mse_loss(synthesized, identity_anchor)
            objective = target_term + lambda_identity * identity_term

            gradients = torch.autograd.grad(
                objective,
                synthesized,
                retain_graph=False,
                create_graph=False,
            )[0]

            synthesized = synthesized - step_size * gradients.sign()
            synthesized = synthesized.clamp(clamp_range[0], clamp_range[1]).detach()
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    return synthesized
