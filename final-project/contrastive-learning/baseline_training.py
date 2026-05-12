from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CIFAR10LTDataset, CreditCardFraudDataset
from src.data.imbalance import analyze_class_distribution, get_class_weights
from src.data.transforms import get_cifar10_transform, get_creditcard_transform
from src.evaluation.metrics import ClassificationMetrics
from src.utils.seed import set_seed

import argparse

# =========================================================
# CREDIT CARD BASELINE (LOGISTIC REGRESSION)
# =========================================================
def train_creditcard_baseline(seed: int = 42) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = CreditCardFraudDataset(
        split="train",
        transform=get_creditcard_transform(train=True, dropout_rate=0.1, noise_mean=0.0, noise_std=0.01),
        normalize=True
    )
    test_dataset = CreditCardFraudDataset(
        split="test",
        transform=get_creditcard_transform(train=False),
        normalize=True
    )

    # Convert to numpy
    X_train = np.array([x.numpy() for x, _ in train_dataset])
    y_train = np.array([y for _, y in train_dataset])

    X_test = np.array([x.numpy() for x, _ in test_dataset])
    y_test = np.array([y for _, y in test_dataset])

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Convert to tensors
    y_pred = torch.tensor(y_pred).long().to(device)
    y_test = torch.tensor(y_test).long().to(device)

    # Metrics
    metrics_calc = ClassificationMetrics(num_classes=2)
    metrics_calc.update(y_pred, y_test)

    overall_metrics = metrics_calc.compute()
    per_class_metrics = metrics_calc.compute_per_class()

    print("\nTest Results:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\nPer-class metrics:")
    for class_id, metrics in per_class_metrics.items():
        print(
            f"\nClass {class_id}: "
            f"- Precision = {metrics['precision']:.4f}, "
            f"- Recall = {metrics['recall']:.4f}, "
            f"- F1 = {metrics['f1']:.4f}"
        )


# =========================================================
# CIFAR10-LT BASELINE (RESNET50)
# =========================================================
def train_cifar10lt_baseline(seed: int = 42, num_epochs: int = 100) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256

    # Dataset
    train_dataset = CIFAR10LTDataset(
        split="train",
        transform=get_cifar10_transform(train=True, image_size=32, horizontal_flip=True, crop_padding=4),
        dataset_config="r-10"
    )
    val_dataset = CIFAR10LTDataset(
        split="val",
        transform=get_cifar10_transform(train=False),
        dataset_config="r-10"
    )
    test_dataset = CIFAR10LTDataset(
        split="test",
        transform=get_cifar10_transform(train=False),
        dataset_config="r-10"
    )

    # Class weights
    class_counts = analyze_class_distribution(train_dataset)
    class_weights = get_class_weights(class_counts, num_classes=10)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    from torchvision.models import resnet50
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 10)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0
    best_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Test
    logits_list, labels_list = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)

            logits_list.append(logits.cpu())
            labels_list.append(y)

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

    metrics_calc = ClassificationMetrics(num_classes=10)
    metrics_calc.update(logits, labels)

    overall_metrics = metrics_calc.compute()
    per_class_metrics = metrics_calc.compute_per_class()

    print("\nTest Results:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\nPer-class metrics:")
    for class_id, metrics in per_class_metrics.items():
        print(
            f"\nClass {class_id}: "
            f"- Precision = {metrics['precision']:.4f}, "
            f"- Recall = {metrics['recall']:.4f}, "
            f"- F1 = {metrics['f1']:.4f}"
        )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["creditcard", "cifar10lt", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset in ["creditcard", "all"]:
        print("\nCredit Card Results:")
        train_creditcard_baseline(seed=args.seed)

    if args.dataset in ["cifar10lt", "all"]:
        print("\nCIFAR10-LT Results:")
        train_cifar10lt_baseline(seed=args.seed)
