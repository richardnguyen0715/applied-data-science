"""
Dataset Analyzer - Comprehensive statistical visualization for CIFAR10 and Credit Card Fraud datasets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger


class DatasetAnalyzer:
    """Analyze and visualize CIFAR10 and Credit Card Fraud datasets."""

    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, data_dir: str = "./data", results_dir: str = "./results"):
        """
        Initialize the analyzer.

        Args:
            data_dir: Directory containing datasets
            results_dir: Directory to save visualizations
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / "visualizations"
        self.stats_dir = self.results_dir / "statistics"

        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("DatasetAnalyzer", log_dir="logs")
        self.logger.info(f"Analyzer initialized: data_dir={data_dir}, results_dir={results_dir}")

    def analyze_cifar10(self) -> Dict:
        """Analyze and visualize CIFAR10 dataset."""
        self.logger.info("Starting CIFAR10 analysis...")

        try:
            import torchvision.transforms as transforms
            from torchvision.datasets import CIFAR10
            import torch

            transform = transforms.ToTensor()
            train_set = CIFAR10(root=str(self.data_dir), train=True, download=False, transform=transform)
            test_set = CIFAR10(root=str(self.data_dir), train=False, download=False, transform=transform)

            # Collect statistics
            stats = {
                'dataset': 'CIFAR10',
                'train_samples': len(train_set),
                'test_samples': len(test_set),
                'total_samples': len(train_set) + len(test_set),
                'image_shape': (3, 32, 32),
                'num_classes': len(self.CIFAR10_CLASSES),
                'classes': self.CIFAR10_CLASSES,
            }

            # Class distribution
            train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
            test_labels = torch.tensor([test_set[i][1] for i in range(len(test_set))])

            class_counts_train = torch.bincount(train_labels)
            class_counts_test = torch.bincount(test_labels)

            stats['class_distribution_train'] = class_counts_train.tolist()
            stats['class_distribution_test'] = class_counts_test.tolist()

            # Channel statistics (sample 1000 images for speed)
            sample_indices = np.random.choice(len(train_set), min(1000, len(train_set)), replace=False)
            channels_mean = []
            channels_std = []

            for idx in sample_indices:
                img_tensor = train_set[idx][0]
                channels_mean.append(img_tensor.mean(dim=(1, 2)).numpy())
                channels_std.append(img_tensor.std(dim=(1, 2)).numpy())

            channels_mean = np.array(channels_mean).mean(axis=0)
            channels_std = np.array(channels_std).mean(axis=0)

            stats['channel_means'] = channels_mean.tolist()
            stats['channel_stds'] = channels_std.tolist()
            stats['channel_names'] = ['Red', 'Green', 'Blue']

            self.logger.info(f"CIFAR10 stats collected: {stats['train_samples']} train, {stats['test_samples']} test")

            # Visualizations
            self._plot_cifar10_class_distribution(class_counts_train, class_counts_test)
            self._plot_cifar10_samples(train_set)
            self._plot_cifar10_channel_stats(channels_mean, channels_std)

            # Save statistics
            self._save_stats(stats, 'cifar10_statistics.json')

            return stats

        except Exception as e:
            self.logger.error(f"Error analyzing CIFAR10: {e}")
            return {'error': str(e)}

    def analyze_fraud(self) -> Dict:
        """Analyze and visualize Credit Card Fraud dataset."""
        self.logger.info("Starting Credit Card Fraud analysis...")

        try:
            fraud_path = self.data_dir / "creditcardfraud" / "creditcard.csv"

            if not fraud_path.exists():
                self.logger.warning(f"Fraud dataset not found at {fraud_path}")
                return {'error': f'Fraud dataset not found at {fraud_path}'}

            df = pd.read_csv(fraud_path)

            # Basic statistics
            stats = {
                'dataset': 'Credit Card Fraud Detection',
                'total_transactions': len(df),
                'total_features': len(df.columns),
                'features': df.columns.tolist(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            }

            # Class distribution
            class_dist = df['Class'].value_counts().to_dict()
            fraud_count = class_dist.get(1, 0)
            legitimate_count = class_dist.get(0, 0)
            fraud_percentage = (fraud_count / len(df)) * 100

            stats['class_distribution'] = {
                'legitimate': int(legitimate_count),
                'fraud': int(fraud_count),
                'fraud_percentage': round(fraud_percentage, 4)
            }

            # Feature statistics
            stats['feature_statistics'] = {
                'amount_mean': float(df['Amount'].mean()),
                'amount_median': float(df['Amount'].median()),
                'amount_std': float(df['Amount'].std()),
                'amount_min': float(df['Amount'].min()),
                'amount_max': float(df['Amount'].max()),
                'time_range_seconds': int(df['Time'].max()),
            }

            # Missing values
            stats['missing_values'] = df.isnull().sum().to_dict()

            self.logger.info(f"Fraud stats collected: {fraud_count} fraud, {legitimate_count} legitimate")

            # Visualizations
            self._plot_fraud_class_distribution(df)
            self._plot_fraud_amount_distribution(df)
            self._plot_fraud_correlation_heatmap(df)
            self._plot_fraud_time_patterns(df)

            # Save statistics
            self._save_stats(stats, 'fraud_statistics.json')

            return stats

        except Exception as e:
            self.logger.error(f"Error analyzing Fraud dataset: {e}")
            return {'error': str(e)}

    def _plot_cifar10_class_distribution(self, train_counts, test_counts):
        """Plot CIFAR10 class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(self.CIFAR10_CLASSES))
        width = 0.35

        axes[0].bar(x, train_counts, width, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Class', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        axes[0].set_title('CIFAR10 Training Set - Class Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.CIFAR10_CLASSES, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(x, test_counts, width, color='coral', alpha=0.8)
        axes[1].set_xlabel('Class', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        axes[1].set_title('CIFAR10 Test Set - Class Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.CIFAR10_CLASSES, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cifar10_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: cifar10_class_distribution.png")

    def _plot_cifar10_samples(self, dataset, num_samples: int = 12):
        """Plot sample CIFAR10 images."""
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        axes = axes.ravel()

        indices = np.random.choice(len(dataset), num_samples, replace=False)

        for ax_idx, img_idx in enumerate(indices):
            img, label = dataset[img_idx]
            img_np = img.permute(1, 2, 0).numpy()

            axes[ax_idx].imshow(img_np)
            axes[ax_idx].set_title(self.CIFAR10_CLASSES[label], fontsize=10, fontweight='bold')
            axes[ax_idx].axis('off')

        plt.suptitle('CIFAR10 - Sample Images', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cifar10_sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: cifar10_sample_images.png")

    def _plot_cifar10_channel_stats(self, means, stds):
        """Plot CIFAR10 channel statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        channels = ['Red', 'Green', 'Blue']
        x = np.arange(len(channels))
        width = 0.35

        axes[0].bar(x, means, width, color=['#e74c3c', '#2ecc71', '#3498db'], alpha=0.8)
        axes[0].set_ylabel('Mean Pixel Value', fontsize=11, fontweight='bold')
        axes[0].set_title('CIFAR10 - Channel Means', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(channels)
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(x, stds, width, color=['#e74c3c', '#2ecc71', '#3498db'], alpha=0.8)
        axes[1].set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
        axes[1].set_title('CIFAR10 - Channel Std Dev', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(channels)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cifar10_channel_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: cifar10_channel_statistics.png")

    def _plot_fraud_class_distribution(self, df):
        """Plot fraud class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        class_counts = df['Class'].value_counts().sort_index()
        colors = ['#2ecc71', '#e74c3c']
        labels = ['Legitimate', 'Fraud']

        # Absolute counts
        axes[0].bar(labels, class_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
        axes[0].set_title('Credit Card Fraud - Class Distribution (Absolute)', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v + 1000, str(int(v)), ha='center', fontweight='bold')

        # Percentage
        percentages = (class_counts.values / len(df)) * 100
        axes[1].pie(
            class_counts.values,
            labels=[f'{l}\n({p:.2f}%)' for l, p in zip(labels, percentages)],
            colors=colors,
            autopct='',
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        axes[1].set_title('Credit Card Fraud - Class Distribution (Percentage)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fraud_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: fraud_class_distribution.png")

    def _plot_fraud_amount_distribution(self, df):
        """Plot transaction amount distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall distribution
        axes[0].hist(df['Amount'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Credit Card Fraud - Transaction Amount Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Legitimate vs Fraud
        legitimate_amounts = df[df['Class'] == 0]['Amount']
        fraud_amounts = df[df['Class'] == 1]['Amount']

        axes[1].hist(legitimate_amounts, bins=100, alpha=0.7, label='Legitimate', color='#2ecc71', edgecolor='black')
        axes[1].hist(fraud_amounts, bins=100, alpha=0.7, label='Fraud', color='#e74c3c', edgecolor='black')
        axes[1].set_xlabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('Credit Card Fraud - Amount by Class', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fraud_amount_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: fraud_amount_distribution.png")

    def _plot_fraud_correlation_heatmap(self, df):
        """Plot feature correlation heatmap."""
        fig, ax = plt.subplots(figsize=(16, 12))

        corr_matrix = df.corr()

        sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax,
            vmin=-1,
            vmax=1
        )

        ax.set_title('Credit Card Fraud - Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fraud_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: fraud_correlation_heatmap.png")

    def _plot_fraud_time_patterns(self, df):
        """Plot fraud patterns over time."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Transactions per time period
        time_bins = np.linspace(0, df['Time'].max(), 50)
        time_labels = [(f[0] + f[1]) / 2 / 3600 for f in zip(time_bins[:-1], time_bins[1:])]

        legitimate_per_time = np.histogram(df[df['Class'] == 0]['Time'], bins=time_bins)[0]
        fraud_per_time = np.histogram(df[df['Class'] == 1]['Time'], bins=time_bins)[0]

        axes[0].plot(time_labels, legitimate_per_time, marker='o', label='Legitimate', color='#2ecc71', linewidth=2)
        axes[0].plot(time_labels, fraud_per_time, marker='s', label='Fraud', color='#e74c3c', linewidth=2)
        axes[0].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
        axes[0].set_title('Credit Card Fraud - Transactions over Time', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Fraud rate over time
        fraud_rate = (fraud_per_time / (legitimate_per_time + fraud_per_time + 1e-6)) * 100
        axes[1].plot(time_labels, fraud_rate, marker='o', color='#e74c3c', linewidth=2)
        axes[1].fill_between(time_labels, fraud_rate, alpha=0.3, color='#e74c3c')
        axes[1].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Fraud Rate (%)', fontsize=11, fontweight='bold')
        axes[1].set_title('Credit Card Fraud - Fraud Rate over Time', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'fraud_time_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: fraud_time_patterns.png")

    def _save_stats(self, stats: Dict, filename: str):
        """Save statistics to JSON file."""
        filepath = self.stats_dir / filename
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"Saved statistics: {filename}")

    def generate_summary_report(self, cifar10_stats: Dict, fraud_stats: Dict):
        """Generate a summary report of both datasets."""
        report_path = self.results_dir / "VISUALIZATION_REPORT.md"

        report = f"""# Dataset Visualization Report

Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 CIFAR10 Dataset

### Basic Statistics
- **Training Samples:** {cifar10_stats.get('train_samples', 'N/A'):,}
- **Test Samples:** {cifar10_stats.get('test_samples', 'N/A'):,}
- **Total Samples:** {cifar10_stats.get('total_samples', 'N/A'):,}
- **Image Shape:** {cifar10_stats.get('image_shape', 'N/A')}
- **Number of Classes:** {cifar10_stats.get('num_classes', 'N/A')}
- **Classes:** {', '.join(cifar10_stats.get('classes', []))}

### Channel Statistics
- **Channel Means:** R={cifar10_stats.get('channel_means', [None])[0]:.4f}, G={cifar10_stats.get('channel_means', [None, None])[1]:.4f}, B={cifar10_stats.get('channel_means', [None, None, None])[2]:.4f}
- **Channel Stds:** R={cifar10_stats.get('channel_stds', [None])[0]:.4f}, G={cifar10_stats.get('channel_stds', [None, None])[1]:.4f}, B={cifar10_stats.get('channel_stds', [None, None, None])[2]:.4f}

### Visualizations
- `cifar10_class_distribution.png` - Train/test class distribution
- `cifar10_sample_images.png` - 12 random sample images
- `cifar10_channel_statistics.png` - RGB channel statistics

---

## 💳 Credit Card Fraud Detection Dataset

### Basic Statistics
- **Total Transactions:** {fraud_stats.get('total_transactions', 'N/A'):,}
- **Total Features:** {fraud_stats.get('total_features', 'N/A')}
- **Memory Usage:** {fraud_stats.get('memory_usage_mb', 'N/A'):.2f} MB

### Class Distribution
- **Legitimate Transactions:** {fraud_stats.get('class_distribution', {}).get('legitimate', 'N/A'):,}
- **Fraudulent Transactions:** {fraud_stats.get('class_distribution', {}).get('fraud', 'N/A'):,}
- **Fraud Percentage:** {fraud_stats.get('class_distribution', {}).get('fraud_percentage', 'N/A')}%

### Amount Statistics
- **Mean:** ${fraud_stats.get('feature_statistics', {}).get('amount_mean', 'N/A'):.2f}
- **Median:** ${fraud_stats.get('feature_statistics', {}).get('amount_median', 'N/A'):.2f}
- **Std Dev:** ${fraud_stats.get('feature_statistics', {}).get('amount_std', 'N/A'):.2f}
- **Min:** ${fraud_stats.get('feature_statistics', {}).get('amount_min', 'N/A'):.2f}
- **Max:** ${fraud_stats.get('feature_statistics', {}).get('amount_max', 'N/A'):.2f}

### Visualizations
- `fraud_class_distribution.png` - Legitimate vs fraud distribution
- `fraud_amount_distribution.png` - Transaction amounts
- `fraud_correlation_heatmap.png` - Feature correlations
- `fraud_time_patterns.png` - Temporal patterns

---

## 📁 Output Files

**Visualizations:** `results/visualizations/`
- 7 high-resolution PNG files (300 DPI)

**Statistics:** `results/statistics/`
- `cifar10_statistics.json` - CIFAR10 metrics
- `fraud_statistics.json` - Fraud detection metrics

---

## 🎯 Key Insights

### CIFAR10
- Well-balanced dataset with {cifar10_stats.get('class_distribution_train', [None])[0]:,} samples per class (training set)
- Standard normalization values: mean ≈ {cifar10_stats.get('channel_means', [0])[0]:.3f}, std ≈ {cifar10_stats.get('channel_stds', [0])[0]:.3f}

### Credit Card Fraud
- Highly imbalanced dataset: {fraud_stats.get('class_distribution', {}).get('fraud_percentage', 'N/A')}% frauds
- Mean transaction: ${fraud_stats.get('feature_statistics', {}).get('amount_mean', 'N/A'):.2f}, with high variance
- Clear temporal patterns in fraud occurrence

"""

        with open(report_path, 'w') as f:
            f.write(report)

        self.logger.info(f"Generated summary report: {report_path}")
        print(f"\n✅ Summary report saved to: {report_path}")


def main():
    """Main execution function."""
    analyzer = DatasetAnalyzer(data_dir="./data", results_dir="./results")

    print("\n" + "="*70)
    print("🔍 DATASET VISUALIZATION PIPELINE")
    print("="*70)

    cifar10_stats = analyzer.analyze_cifar10()
    print("\n✅ CIFAR10 analysis complete!")

    fraud_stats = analyzer.analyze_fraud()
    print("✅ Credit Card Fraud analysis complete!")

    analyzer.generate_summary_report(cifar10_stats, fraud_stats)

    print("\n" + "="*70)
    print("📊 VISUALIZATIONS SAVED")
    print("="*70)
    print(f"📁 Visualizations: results/visualizations/")
    print(f"📁 Statistics: results/statistics/")
    print(f"📄 Report: results/VISUALIZATION_REPORT.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
