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
from src.utils.sampler import BalancedBatchSampler
from src.utils.seed import set_seed

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import argparse

# =========================================================
# CIFAR10-LT BASELINE (RESNET50)
# =========================================================
def train_cifar10lt_baseline(seed: int = 42, num_epochs: int = 200):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256
    base_lr = 0.02
    warmup_epochs = 5

    # ======================
    # DATASET
    # ======================
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

    # ======================
    # CLASS WEIGHTS
    # ======================
    class_counts = analyze_class_distribution(train_dataset)

    class_weights = get_class_weights(class_counts, num_classes=10)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # ======================
    # SAMPLER
    # ======================
    batch_sampler = BalancedBatchSampler(
        labels=train_dataset.labels,
        n_classes=10,
        n_samples=8
    )

    # ======================
    # DATALOADER
    # ======================
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ======================
    # MODEL
    # ======================
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    model.to(device)

    # ======================
    # OPTIMIZER & LOSS
    # ======================

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)

    # Scheduler
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_epochs
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0
    best_state = None

    # ======================
    # TRAINING LOOP
    # ======================
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ======================
        # VALIDATION
        # ======================
        model.eval()
        val_loss = 0
        correct = 0
        total_val = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total_val += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total_val

        print(
            f"Epoch {epoch+1} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ===== Save best =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # ======================
    # TEST
    # ======================
    model.load_state_dict(best_state)
    model.eval()

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
# CREDIT CARD BASELINE (LOGISTIC REGRESSION)
# =========================================================
def train_creditcard_baseline(seed: int = 42) -> Dict[str, float]:
    set_seed(seed)

    # Load data
    train_dataset = CreditCardFraudDataset(
        split="train",
        normalize=False
    )
    test_dataset = CreditCardFraudDataset(
        split="test",
        normalize=False
    )

    # Convert to numpy
    X_train = np.array([x.numpy() for x, _ in train_dataset])
    y_train = np.array([y for _, y in train_dataset])

    X_test = np.array([x.numpy() for x, _ in test_dataset])
    y_test = np.array([y for _, y in test_dataset])

    # Standardize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    # Overall metrics
    print("\nTest Results:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    # Per-class metrics
    print("\nPer-class metrics:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["creditcard", "cifar10lt", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset in ["cifar10lt", "all"]:
        print("CIFAR10-LT Results:")
        train_cifar10lt_baseline(seed=args.seed)
    
    if args.dataset in ["creditcard", "all"]:
        print("Credit Card Results:")
        train_creditcard_baseline(seed=args.seed)


