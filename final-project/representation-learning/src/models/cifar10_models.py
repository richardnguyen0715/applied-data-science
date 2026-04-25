"""CIFAR10 model definitions"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18CIFAR10(nn.Module):
    """ResNet18 adapted for CIFAR10"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        # Adapt for CIFAR10 (32x32 images)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR10"""
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
