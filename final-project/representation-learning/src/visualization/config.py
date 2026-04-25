"""
Visualization Configuration Module
Customizable settings for dataset analysis
"""

import json
from pathlib import Path
from typing import Dict, Any

# Default visualization parameters
DEFAULT_CONFIG = {
    "cifar10": {
        "num_sample_images": 12,
        "sample_grid": (3, 4),
        "channels": ["Red", "Green", "Blue"],
        "dpi": 300,
        "figure_formats": ["png"],
    },
    "fraud": {
        "amount_bins": 100,
        "time_bins": 50,
        "correlation_figsize": (16, 12),
        "heatmap_cmap": "coolwarm",
        "dpi": 300,
        "figure_formats": ["png"],
    },
    "output": {
        "visualizations_dir": "results/visualizations",
        "statistics_dir": "results/statistics",
        "report_name": "VISUALIZATION_REPORT.md",
        "log_file": "logs/analyzer.log",
    },
    "plotting": {
        "style": "seaborn-v0_8-darkgrid",
        "font_size": 10,
        "title_size": 12,
        "label_size": 11,
        "colors": {
            "cifar_train": "steelblue",
            "cifar_test": "coral",
            "fraud_legitimate": "#2ecc71",
            "fraud_fraud": "#e74c3c",
            "rgb_red": "#e74c3c",
            "rgb_green": "#2ecc71",
            "rgb_blue": "#3498db",
        },
    },
}


class VisualizationConfig:
    """Manage visualization configuration settings."""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to JSON config file (optional)
        """
        self.config = DEFAULT_CONFIG.copy()

        if config_file and Path(config_file).exists():
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, "r") as f:
            custom_config = json.load(f)
            self._merge_config(custom_config)

    def _merge_config(self, custom_config: Dict[str, Any]):
        """Recursively merge custom config with defaults."""
        for key, value in custom_config.items():
            if key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if section in self.config:
            return self.config[section].get(key, default)
        return default

    def save_config(self, config_file: str):
        """Save configuration to JSON file."""
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.config.copy()


def get_default_config() -> Dict:
    """Get default configuration dictionary."""
    return DEFAULT_CONFIG.copy()


def create_sample_config(output_file: str = "visualization_config.json"):
    """Create sample configuration file."""
    config = get_default_config()
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Sample config created: {output_file}")
    return config
