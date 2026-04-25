"""CIFAR10 data preprocessing and loading"""

import torch
import kagglehub
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import numpy as np


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
        
        # Download dataset using kagglehub
        path = kagglehub.dataset_download("akhiltheerthala/imbalanced-cifar-10")
        
        # Load the dataset from the downloaded path
        train_dataset = self._load_cifar10_dataset(path, train=True)
        test_dataset = self._load_cifar10_dataset(path, train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, test_loader
    
    def _load_cifar10_dataset(self, path: str, train: bool = True):
        """Load CIFAR10 dataset from kagglehub path and apply transforms"""
        split = 'train' if train else 'test'
        data_file = os.path.join(path, f'{split}_data.npy')
        label_file = os.path.join(path, f'{split}_labels.npy')
        
        if os.path.exists(data_file) and os.path.exists(label_file):
            images = np.load(data_file)
            labels = np.load(label_file)
        else:
            # Fallback: try loading from pickle files if available
            images, labels = self._load_from_pickle(path, split)
        
        # Convert to tensors and apply transforms
        images = torch.from_numpy(images).float() / 255.0
        labels = torch.from_numpy(labels).long()
        
        class TransformDataset(TensorDataset):
            def __init__(self, images, labels, transform=None):
                super().__init__(images, labels)
                self.transform = transform
            
            def __getitem__(self, idx):
                img, label = super().__getitem__(idx)
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        transforms_to_use = self.train_transforms if train else self.val_transforms
        return TransformDataset(images, labels, transform=transforms_to_use)
    
    def _load_from_pickle(self, path: str, split: str):
        """Load CIFAR10 data from pickle files"""
        if split == 'train':
            data_batches = [os.path.join(path, f'data_batch_{i}') for i in range(1, 6)]
        else:
            data_batches = [os.path.join(path, 'test_batch')]
        
        images_list = []
        labels_list = []
        
        for batch_file in data_batches:
            if os.path.exists(batch_file):
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    images_list.append(batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                    labels_list.append(np.array(batch[b'labels']))
        
        if images_list:
            images = np.concatenate(images_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            return images, labels
        else:
            raise FileNotFoundError(f"No data files found in {path}")
