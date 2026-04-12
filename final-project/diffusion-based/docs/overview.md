# Project Overview

## Architecture

This project demonstrates a complete machine learning system for handling class imbalance using diffusion models. The architecture consists of four main components:

### 1. Data Pipeline
- **CIFAR-10-LT Loading**: HuggingFace dataset integration
- **Transformations**: Augmentation and normalization
- **Imbalance Analysis**: Distribution statistics and visualization

### 2. Diffusion Model Component
- **DDPM**: Conditional Denoising Diffusion Probabilistic Model
- **U-Net**: Deep residual network with attention mechanisms
- **Scheduler**: Noise schedule management
- **EMA**: Exponential moving average for better convergence

### 3. Classification Component
- **Models**: ResNet18 and lightweight ConvNet
- **Training**: Full supervised learning pipeline
- **Evaluation**: Comprehensive metrics and visualizations

### 4. Sampling & Balancing
- **Synthetic Generation**: Create minority class samples
- **Dataset Balancing**: Combine real and synthetic data
- **Pipeline Orchestration**: End-to-end workflow management

## Design Patterns

### Modular Architecture
- **Separation of Concerns**: Each module has a single responsibility
- **Dependency Injection**: Configuration passed to components
- **Abstract Base Classes**: Common patterns across models

### Reproducibility
- **Fixed Random Seeds**: All randomness is controlled
- **Configuration Management**: YAML files for hyperparameters
- **Logging**: Detailed execution logs for debugging

### Type Safety
- **Full Type Hints**: PEP 484 compliance
- **Dataclasses**: Strong typing for configurations
- **Runtime Checks**: Assertion and validation

## Data Flow

```
Raw CIFAR-10-LT
        ↓
    Transform
        ↓
Imbalanced Dataset (Original)
        ├─→ Diffusion Model Training
        │        ↓
        │   Synthetic Generation
        │        ↓
        └─→ Dataset Balancing
                   ↓
          Balanced Dataset
                   ↓
          Classifier Training
                   ↓
          Evaluation & Metrics
```

## Model Hierarchy

```
BaseModel
├── DiffusionModel
│   └── UNet
│       ├── ResidualBlock
│       └── AttentionBlock
└── Classifier
    ├── ResNet18Classifier
    └── LightweightConvNet
```

## Key Technologies

- **PyTorch 2.0+**: Deep learning framework
- **Torchvision**: Computer vision utilities
- **HuggingFace Datasets**: Data loading
- **NumPy/SciPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization
- **YAML**: Configuration management
- **Python Logging**: Execution logging

## Configuration System

The project uses a tiered configuration system:

1. **Default Values**: Defined in dataclass constructors
2. **YAML Files**: Override defaults in configs/
3. **Runtime Parameters**: Arguments to pipeline functions
4. **Environment Variables**: System-level settings

This allows flexible configuration at multiple levels.

## Logging Strategy

- **Module-Level Loggers**: Each module has its own logger
- **Centralized Configuration**: Setup via utils/logger.py
- **File + Console Output**: Logs saved and displayed
- **Structured Format**: Timestamp, level, module, message

## Performance Considerations

### Training
- **Batch Processing**: Efficient GPU utilization
- **Gradient Clipping**: Stable training
- **Learning Rate Scheduling**: Cosine annealing
- **EMA**: Better model convergence

### Inference
- **Model Evaluation Mode**: Dropout and batch norm disabled
- **No Gradient Tracking**: torch.no_grad() for efficiency
- **Batch Processing**: Process multiple samples together

### Memory
- **Efficient Data Loading**: pin_memory and num_workers
- **Gradient Accumulation**: Effective batch size increase
- **Checkpoint Management**: Regular checkpoint saving
