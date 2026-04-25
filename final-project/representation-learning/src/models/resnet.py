"""ResNet definitions for CIFAR-10 classification."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet18_Weights, resnet18


class CIFARResNet18(nn.Module):
    """ResNet-18 backbone adapted to 32x32 CIFAR-10 inputs."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        """Initialize a CIFAR-friendly ResNet-18 model.

        Args:
            num_classes: Number of output classes.
            pretrained: Whether to load ImageNet pretrained weights.
        """
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            inputs: Input image batch with shape (N, 3, 32, 32).

        Returns:
            Logits tensor with shape (N, num_classes).
        """
        return self.backbone(inputs)


def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> CIFARResNet18:
    """Factory helper for CIFAR ResNet-18."""
    return CIFARResNet18(num_classes=num_classes, pretrained=pretrained)
