"""Sanity tests for the CIFAR-10 M2m training pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.engine.evaluator import Evaluator
from src.engine.trainer import M2MTrainer
from src.m2m.synthesis import synthesize_m2m
from src.models.resnet import CIFARResNet18
from src.utils.logger import setup_logger


class TestM2MPipeline(unittest.TestCase):
    """Validate core M2m pipeline behavior."""

    def setUp(self) -> None:
        """Initialize shared test fixtures."""
        self.device = torch.device("cpu")
        self.model = CIFARResNet18(num_classes=10, pretrained=False).to(self.device)

        self.images = torch.rand(8, 3, 32, 32)
        self.labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3], dtype=torch.long)

    def test_forward_pass(self) -> None:
        """Model should return logits with shape (N, 10)."""
        logits = self.model(self.images)
        self.assertEqual(tuple(logits.shape), (8, 10))

    def test_m2m_synthesis_shape(self) -> None:
        """M2m synthesis should preserve the input tensor shape."""
        source_images = self.images[:4]
        target_labels = torch.full((4,), 7, dtype=torch.long)

        self.model.train()
        synthesized = synthesize_m2m(
            model=self.model,
            source_images=source_images,
            target_labels=target_labels,
            steps=2,
            step_size=0.005,
            lambda_identity=0.1,
        )

        self.assertEqual(tuple(synthesized.shape), tuple(source_images.shape))
        self.assertTrue(self.model.training)

    def test_one_training_step(self) -> None:
        """One training epoch should run and return finite metrics."""
        dataset = TensorDataset(self.images, self.labels)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = {
                "dataset": {"num_classes": 10},
                "warmup_epochs": 0,
                "training": {
                    "epochs": 1,
                    "amp": False,
                    "log_interval": 100,
                    "max_synth_per_batch": 2,
                    "checkpoint_dir": str(tmp_path / "checkpoints"),
                },
                "m2m": {
                    "steps": 1,
                    "lr": 0.001,
                    "lambda": 0.1,
                },
            }

            logger = setup_logger(
                name="m2m_test",
                log_dir=str(tmp_path / "logs"),
                log_file="test.log",
            )
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            evaluator = Evaluator(num_classes=10, device=self.device)
            trainer = M2MTrainer(
                model=self.model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                evaluator=evaluator,
                config=config,
                device=self.device,
                logger=logger,
                scheduler=None,
            )

            metrics = trainer.train_one_epoch(epoch=0)

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("num_synthesized", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))


if __name__ == "__main__":
    unittest.main()
