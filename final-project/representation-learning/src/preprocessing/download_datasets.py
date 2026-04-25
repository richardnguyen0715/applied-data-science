"""Dataset download and verification script"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from typing import Tuple
import argparse

import torch
import torchvision.datasets as datasets
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handle downloading and verifying datasets"""
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / 'data'
        self.cifar10_dir = self.data_dir / 'cifar10'
        self.fraud_dir = self.data_dir / 'credit_card_fraud'
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cifar10_dir.mkdir(parents=True, exist_ok=True)
        self.fraud_dir.mkdir(parents=True, exist_ok=True)
    
    def download_cifar10(self) -> bool:
        """Download CIFAR10 dataset"""
        logger.info("=" * 60)
        logger.info("Downloading CIFAR10 Dataset")
        logger.info("=" * 60)
        
        try:
            logger.info(f"Target directory: {self.cifar10_dir}")
            logger.info("Downloading CIFAR10 training set...")
            
            # Download training set
            train_dataset = datasets.CIFAR10(
                root=str(self.cifar10_dir),
                train=True,
                download=True
            )
            logger.info(f"✓ Training set downloaded: {len(train_dataset)} samples")
            
            # Download test set
            logger.info("Downloading CIFAR10 test set...")
            test_dataset = datasets.CIFAR10(
                root=str(self.cifar10_dir),
                train=False,
                download=True
            )
            logger.info(f"✓ Test set downloaded: {len(test_dataset)} samples")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to download CIFAR10: {e}")
            return False
    
    def verify_cifar10(self) -> bool:
        """Verify CIFAR10 dataset integrity"""
        logger.info("Verifying CIFAR10 dataset...")
        
        try:
            # Try to load datasets
            train_dataset = datasets.CIFAR10(
                root=str(self.cifar10_dir),
                train=True,
                download=False
            )
            test_dataset = datasets.CIFAR10(
                root=str(self.cifar10_dir),
                train=False,
                download=False
            )
            
            # Verify sizes
            assert len(train_dataset) == 50000, f"Expected 50000 training samples, got {len(train_dataset)}"
            assert len(test_dataset) == 10000, f"Expected 10000 test samples, got {len(test_dataset)}"
            
            # Check sample
            img, label = train_dataset[0]
            assert img.size == (32, 32), f"Expected image size (32, 32), got {img.size}"
            assert 0 <= label < 10, f"Expected label 0-9, got {label}"
            
            logger.info("✓ CIFAR10 verification passed")
            logger.info(f"  - Training samples: 50,000")
            logger.info(f"  - Test samples: 10,000")
            logger.info(f"  - Image size: 32x32")
            logger.info(f"  - Number of classes: 10")
            
            return True
        except Exception as e:
            logger.error(f"✗ CIFAR10 verification failed: {e}")
            return False
    
    def setup_fraud_dataset(self) -> bool:
        """Setup credit card fraud dataset (requires manual download)"""
        logger.info("=" * 60)
        logger.info("Credit Card Fraud Detection Dataset Setup")
        logger.info("=" * 60)
        
        csv_path = self.fraud_dir / 'creditcard.csv'
        
        if csv_path.exists():
            logger.info(f"✓ Dataset file found: {csv_path}")
            return self.verify_fraud_dataset()
        else:
            logger.warning("✗ Dataset file not found")
            logger.info("\nTo download the dataset:")
            logger.info("1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            logger.info("2. Download creditcard.csv")
            logger.info(f"3. Place it in: {self.fraud_dir}")
            logger.info("\nOr use Kaggle CLI:")
            logger.info("  kaggle datasets download -d mlg-ulb/creditcardfraud")
            logger.info(f"  unzip creditcardfraud.zip -d {self.fraud_dir}")
            logger.info("  rm creditcardfraud.zip")
            return False
    
    def verify_fraud_dataset(self) -> bool:
        """Verify credit card fraud dataset integrity"""
        logger.info("Verifying Credit Card Fraud dataset...")
        
        try:
            csv_path = self.fraud_dir / 'creditcard.csv'
            
            if not csv_path.exists():
                logger.error(f"✗ File not found: {csv_path}")
                return False
            
            # Load and verify
            df = pd.read_csv(csv_path)
            
            # Verify structure
            assert 'Class' in df.columns, "Missing 'Class' column"
            assert len(df) > 0, "Dataset is empty"
            assert df.shape[1] >= 30, f"Expected at least 30 features, got {df.shape[1]}"
            
            # Check for data quality
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            assert missing_ratio < 0.01, f"Too many missing values: {missing_ratio:.2%}"
            
            # Verify target distribution
            class_dist = df['Class'].value_counts()
            logger.info("✓ Credit Card Fraud dataset verification passed")
            logger.info(f"  - Total transactions: {len(df):,}")
            logger.info(f"  - Number of features: {df.shape[1]}")
            logger.info(f"  - Legitimate transactions: {class_dist.get(0, 0):,}")
            logger.info(f"  - Fraudulent transactions: {class_dist.get(1, 0):,}")
            logger.info(f"  - Fraud rate: {(class_dist.get(1, 0) / len(df) * 100):.2f}%")
            logger.info(f"  - Missing values: {missing_ratio:.2%}")
            
            return True
        except FileNotFoundError:
            logger.error(f"✗ File not found: {csv_path}")
            return False
        except Exception as e:
            logger.error(f"✗ Verification failed: {e}")
            return False
    
    def generate_summary(self) -> dict:
        """Generate dataset summary"""
        summary = {
            'cifar10': {
                'status': 'unknown',
                'location': str(self.cifar10_dir),
                'details': {}
            },
            'credit_card_fraud': {
                'status': 'unknown',
                'location': str(self.fraud_dir),
                'details': {}
            }
        }
        
        # Check CIFAR10
        try:
            train = datasets.CIFAR10(str(self.cifar10_dir), train=True, download=False)
            test = datasets.CIFAR10(str(self.cifar10_dir), train=False, download=False)
            summary['cifar10']['status'] = 'ready'
            summary['cifar10']['details'] = {
                'train_samples': len(train),
                'test_samples': len(test),
                'total_samples': len(train) + len(test),
                'image_size': '32x32',
                'channels': 3,
                'classes': 10
            }
        except:
            summary['cifar10']['status'] = 'missing'
        
        # Check fraud dataset
        csv_path = self.fraud_dir / 'creditcard.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                summary['credit_card_fraud']['status'] = 'ready'
                summary['credit_card_fraud']['details'] = {
                    'transactions': len(df),
                    'features': df.shape[1],
                    'fraud_cases': int(df['Class'].sum()),
                    'fraud_rate': f"{(df['Class'].sum() / len(df) * 100):.2f}%"
                }
            except:
                summary['credit_card_fraud']['status'] = 'corrupted'
        else:
            summary['credit_card_fraud']['status'] = 'missing'
        
        return summary
    
    def run(self, skip_cifar10: bool = False) -> Tuple[bool, dict]:
        """Run full pipeline"""
        logger.info("\n" + "=" * 60)
        logger.info("DATASET SETUP PIPELINE")
        logger.info("=" * 60 + "\n")
        
        success = True
        
        # CIFAR10
        if not skip_cifar10:
            if not self.download_cifar10():
                success = False
            if not self.verify_cifar10():
                success = False
        else:
            logger.info("Skipping CIFAR10 download (already exists)")
            if not self.verify_cifar10():
                success = False
        
        logger.info("")
        
        # Credit Card Fraud
        if not self.setup_fraud_dataset():
            success = False
        if not self.verify_fraud_dataset():
            success = False
        
        # Summary
        logger.info("=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        
        summary = self.generate_summary()
        
        for dataset_name, dataset_info in summary.items():
            logger.info(f"{dataset_name.replace('_', ' ').upper()}")
            logger.info(f"  Status: {dataset_info['status'].upper()}")
            logger.info(f"  Location: {dataset_info['location']}")
            if dataset_info['details']:
                for key, value in dataset_info['details'].items():
                    logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)
        if success:
            logger.info("✓ All datasets ready!")
        else:
            logger.warning("⚠ Some datasets need attention (see above)")
        logger.info("=" * 60)
        
        return success, summary


def main():
    parser = argparse.ArgumentParser(description='Download and setup datasets')
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='Project root directory (default: current directory)'
    )
    parser.add_argument(
        '--skip-cifar10',
        action='store_true',
        help='Skip CIFAR10 download if already exists'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets without downloading'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.project_root)
    
    if args.verify_only:
        logger.info("Running verification only...")
        downloader.verify_cifar10()
        downloader.verify_fraud_dataset()
    else:
        success, summary = downloader.run(skip_cifar10=args.skip_cifar10)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
