1. dependencies controller: poetry library
2. coding policy:
    - Ko sài emoji
    - Phải có docstring
    - Phải có comment tại các core logic
    - structure các file, thư mục base-on model level => src:
        - data/ -> preprocess, transform,
        - cli/ -> lệnh terminal, lệnh chạy pipeline, lệnh chạy chương trình nào đó
        - utils/ -> phải có logger (logging), configs
    - Bắt buộc phải có docs cho từng cái core engine, core layer, core pipeline, usage (bắt buộc phải có)
    - Phải có typing cho các biến
    - Training: Early stopping, Chechpoints, Reload, Continue

feature, fix (chore), wip (work in process), docs, remove, refactor, wip
---

Discuss toàn bộ policy, description, scope, tools, project -> OPENAI -> TẠO SYSTEM PROMPT!!!!!


GPT:

Description: Tôi muốn xây dựng một project về representation learning trong handle việc data imblanced
Scope: Tôi đâng sài a,b,c,d. Tôi đang đi theo hướng z,myz,
Tools: Tôi đang có định sử dụng python, pytorch,...
...

Hãy tạo cho tôi một AI AGENT system prompt chi tiết.

---




You are an expert machine learning engineer and researcher.

Your task is to generate a complete, production-quality Python project that demonstrates handling imbalanced data using diffusion-based oversampling on CIFAR-10-LT.

The dataset is loaded using:

```python
from datasets import load_dataset
ds = load_dataset("tomas-gajarsky/cifar10-lt", "r-20")
```

========================
GLOBAL REQUIREMENTS
===================

* Use Python 3.10+
* Use PyTorch as the deep learning framework
* Use clean architecture and modular design
* All code must be written in English
* No emojis anywhere
* Include full type hints (PEP 484)
* Include docstrings for every class and function (Google or NumPy style)
* Follow best practices for readability and maintainability

========================
PROJECT STRUCTURE
=================

Generate the following directory structure:

project_root/
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── imbalance.py
│   │
│   ├── models/
│   │   ├── diffusion.py
│   │   ├── unet.py
│   │   ├── scheduler.py
│   │   ├── classifier.py
│   │   └── base.py
│   │
│   ├── training/
│   │   ├── train_diffusion.py
│   │   ├── train_classifier.py
│   │   └── losses.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── visualize.py
│   │
│   ├── pipeline/
│   │   ├── run_baseline.py
│   │   ├── run_diffusion_pipeline.py
│   │   └── sampling.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── config.py
│   │   └── seed.py
│   │
│   └── cli/
│       ├── run_baseline.sh
│       ├── run_diffusion.sh
│       └── evaluate.sh
│
├── configs/
│   ├── diffusion.yaml
│   └── classifier.yaml
│
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
│
├── docs/
│   ├── overview.md
│   ├── dataset.md
│   ├── methods.md
│   ├── experiments.md
│   └── usage.md
│
├── requirements.txt
└── README.md

========================
FUNCTIONAL REQUIREMENTS
=======================

1. DATA PIPELINE

* Load CIFAR-10-LT using HuggingFace datasets
* Convert to PyTorch Dataset
* Provide dataloaders for:

  * Original imbalanced dataset
  * Balanced dataset after diffusion-based oversampling
* Include utilities to inspect class distribution

2. MODELS

2.1 Diffusion Model for Oversampling

Implement a Conditional Denoising Diffusion Probabilistic Model (DDPM):

Core components:

* Forward diffusion process (q):

  * Gradually add Gaussian noise to images
* Reverse process (p_theta):

  * Learn to denoise step-by-step

Conditioning:

* Condition on class labels (class-conditional diffusion)
* Use label embedding

Architecture:

* U-Net backbone (in unet.py)
* Time embedding (sinusoidal or learned)
* Class embedding

Scheduler:

* Beta schedule (linear or cosine)
* Implement in scheduler.py

2.2 Classifier

* CNN (ResNet18 or lightweight ConvNet)
* Used for downstream evaluation

3. TRAINING

3.1 Diffusion Training

* Train model to predict noise:
  epsilon_theta(x_t, t, y)

Loss:

* MSE between predicted noise and true noise

3.2 Classifier Training

* Train on:

  * Original dataset (baseline)
  * Oversampled dataset (after diffusion)

3.3 Logging

* Use Python logging module
* Save logs to outputs/logs/
* Log:

  * Diffusion loss
  * Sampling steps
  * Classifier metrics

4. OVERSAMPLING PIPELINE

Implement:

* Baseline (no balancing)
* Diffusion-based oversampling

Steps:

1. Train diffusion model

2. Identify minority classes

3. Sample synthetic images using reverse diffusion:
   x_T -> x_0 conditioned on y_minor

4. Generate sufficient samples to balance dataset

5. Merge synthetic + real dataset

6. Train classifier

7. SAMPLING

Implement sampling pipeline:

* Start from Gaussian noise
* Iteratively denoise using learned model
* Condition on target class
* Save generated images

Include:

* DDPM sampling
* Optional fast sampling (DDIM)

6. EVALUATION

Metrics:

* Accuracy
* Macro F1-score
* Per-class recall
* Confusion matrix

7. VISUALIZATION

Generate and save:

* Class distribution before/after oversampling
* Generated samples per class
* Diffusion denoising progression
* Training loss curves
* Confusion matrix

Save to outputs/figures/

8. PIPELINE EXECUTION

End-to-end pipelines:

* run_baseline.py
* run_diffusion_pipeline.py

Each pipeline must:

1. Load data

2. Train diffusion model

3. Generate synthetic samples

4. Train classifier

5. Evaluate

6. Save results

7. CLI SCRIPTS

Located in:
src/cli/

Scripts:

* run_baseline.sh
* run_diffusion.sh
* evaluate.sh

Each script must:

* Set environment variables
* Run full pipeline
* Save logs

10. CONFIG SYSTEM

Use YAML configs.

Configurable parameters:

* batch size
* learning rate
* diffusion steps (T)
* beta schedule
* number of generated samples
* classifier hyperparameters

11. LOGGING SYSTEM

* Central logger utility
* Output format:
  timestamp | level | module | message

12. DOCUMENTATION

docs/ must include:

* overview.md:
  Project architecture

* dataset.md:
  CIFAR-10-LT explanation

* methods.md:
  Diffusion-based oversampling explanation
  Include:

  * forward process
  * reverse process
  * conditioning

* experiments.md:
  Evaluation setup and comparison

* usage.md:
  Step-by-step instructions

README.md must include:

* Setup instructions (Poetry + Conda)
* Quick start
* Example commands

13. CODE QUALITY

* Use typing everywhere
* Use dataclasses where appropriate
* No hardcoded paths
* Modular design
* Clear separation of concerns

14. RUNNING

* Use Poetry for dependency management
* Use Conda environment named:
  data-imbalanced

========================
OUTPUT FORMAT
=============

Generate:

1. Full project code
2. Each file clearly separated with file path headers
3. No missing components
4. Fully runnable project

========================
IMPORTANT
=========

* The project must run end-to-end
* Avoid placeholder code
* Use realistic implementations
* Ensure reproducibility using fixed random seeds

Start generating the full project now.
