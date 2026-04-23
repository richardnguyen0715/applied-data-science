"""Configuration management using dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset_name: str = "cifar10-lt"
    cifar10_config: str = "r-100"
    image_size: int = 32
    num_classes: int = 10
    num_workers: int = 4
    test_size: float = 0.2
    val_size: float = 0.1
    use_smote: bool = False


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation in contrastive learning."""

    horizontal_flip: bool = True
    crop_size: int = 32
    crop_padding: int = 4


@dataclass
class EncoderConfig:
    """Configuration for contrastive encoder model."""

    architecture: str = "resnet18"
    projection_dim: int = 128
    hidden_dim: int = 2048
    num_layers: int = 2
    use_bn: bool = True


@dataclass
class ContrastiveLossConfig:
    """Configuration for contrastive loss."""

    loss_type: str = "nt_xent"
    temperature: float = 0.07
    use_cosine_similarity: bool = True


@dataclass
class ContrastiveTrainingConfig:
    """Configuration for contrastive learning training."""

    batch_size: int = 128
    num_epochs: int = 200
    learning_rate: float = 0.5
    weight_decay: float = 1e-4
    momentum: float = 0.9
    cosine_annealing: bool = True
    warmup_epochs: int = 10


@dataclass
class ClassifierTrainingConfig:
    """Configuration for downstream classifier training."""

    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    patience: int = 20


@dataclass
class ClassifierConfig:
    """Configuration for downstream classifier."""

    num_classes: int = 10
    hidden_dim: Optional[int] = None
    num_layers: int = 2


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    eval_batch_size: int = 128
    num_classes: int = 10
    compute_confusion_matrix: bool = True


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    contrastive_loss: ContrastiveLossConfig = field(default_factory=ContrastiveLossConfig)
    contrastive_training: ContrastiveTrainingConfig = field(default_factory=ContrastiveTrainingConfig)
    classifier_training: ClassifierTrainingConfig = field(default_factory=ClassifierTrainingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "cuda"

    def __post_init__(self):
        """Initialize config after creation."""
        self.classifier.num_classes = self.data.num_classes
        self.evaluation.num_classes = self.data.num_classes
        self.output_dir = Path(self.output_dir)


def load_config_from_yaml(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Config object with values loaded from YAML.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects from YAML
    data_config = DataConfig(**config_dict.get('data', {}))
    augmentation_config = AugmentationConfig(**config_dict.get('augmentation', {}))
    encoder_config = EncoderConfig(**config_dict.get('encoder', {}))
    contrastive_loss_config = ContrastiveLossConfig(**config_dict.get('contrastive_loss', {}))
    contrastive_training_config = ContrastiveTrainingConfig(**config_dict.get('contrastive_training', {}))
    classifier_training_config = ClassifierTrainingConfig(**config_dict.get('classifier_training', {}))
    classifier_config = ClassifierConfig(**config_dict.get('classifier', {}))
    evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
    
    # Create main config
    config = Config(
        data=data_config,
        augmentation=augmentation_config,
        encoder=encoder_config,
        contrastive_loss=contrastive_loss_config,
        contrastive_training=contrastive_training_config,
        classifier_training=classifier_training_config,
        classifier=classifier_config,
        evaluation=evaluation_config,
        seed=config_dict.get('seed', 42),
        output_dir=config_dict.get('output_dir', 'outputs'),
        device=config_dict.get('device', 'cuda'),
    )
    
    return config
