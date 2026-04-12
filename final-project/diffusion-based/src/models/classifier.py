"""Classifier model for downstream evaluation."""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from src.models.base import BaseModel


class ResNet18Classifier(BaseModel):
    """ResNet18-based classifier for CIFAR-10."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        """
        Initialize ResNet18 classifier.

        Args:
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()

        # Load ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Modify for CIFAR-10
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

        # Replace final layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Logits.
        """
        return self.model(x)


class LightweightConvNet(BaseModel):
    """Lightweight convolutional neural network."""

    def __init__(self, num_classes: int = 10) -> None:
        """
        Initialize lightweight ConvNet.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Logits.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_classifier(
    model_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> BaseModel:
    """
    Create a classifier model.

    Args:
        model_name: Name of the model ('resnet18' or 'lightweight').
        num_classes: Number of classes.
        pretrained: Whether to use pretrained weights.

    Returns:
        Classifier model.
    """
    if model_name == "resnet18":
        return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "lightweight":
        return LightweightConvNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
