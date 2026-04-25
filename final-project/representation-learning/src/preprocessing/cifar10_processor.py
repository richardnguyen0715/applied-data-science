"""CIFAR10 data preprocessing and loading"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR10Processor:
    """CIFAR10 dataset processor"""
    
    def __init__(self, config: dict):
        self.config = config
        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()
    
    def _build_train_transforms(self):
        """Build training transforms with augmentation"""
        cfg = self.config['preprocessing']
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(cfg['normalize']['mean'], cfg['normalize']['std']),
        ]
        
        if cfg['augmentation']['enabled']:
            transforms_list = [
                transforms.RandomHorizontalFlip() if cfg['augmentation']['random_flip'] else transforms.Identity(),
                transforms.RandomCrop(32, padding=4) if cfg['augmentation']['random_crop'] else transforms.Identity(),
                *transforms_list,
            ]
        
        return transforms.Compose([t for t in transforms_list if not isinstance(t, transforms.Identity)])
    
    def _build_val_transforms(self):
        """Build validation transforms without augmentation"""
        cfg = self.config['preprocessing']
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['normalize']['mean'], cfg['normalize']['std']),
        ])
    
    def get_dataloaders(self, batch_size: int = None):
        """Get train and test dataloaders"""
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        
        train_dataset = datasets.CIFAR10(
            root=self.config['dataset']['download_dir'],
            train=True,
            download=True,
            transform=self.train_transforms
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.config['dataset']['download_dir'],
            train=False,
            download=True,
            transform=self.val_transforms
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, test_loader
