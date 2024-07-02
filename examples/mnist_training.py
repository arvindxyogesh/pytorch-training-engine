"""examples/mnist_training.py — Demonstrate the training engine on MNIST-like synthetic data."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine.trainer import Trainer
from src.callbacks.checkpoint import ModelCheckpoint
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.logger import CSVLogger, LRSchedulerCallback


def make_synthetic_dataset(n: int = 2000, n_classes: int = 10, seed: int = 42):
    """Synthetic (n, 28, 28) image classification dataset."""
    torch.manual_seed(seed)
    X = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, n_classes, (n,))
    split = int(n * 0.8)
    return (
        TensorDataset(X[:split], y[:split]),
        TensorDataset(X[split:], y[split:]),
    )


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def main():
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    train_ds, val_ds = make_synthetic_dataset(n=2000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256)

    model     = SimpleCNN(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=30,
        grad_clip=1.0,
        callbacks=[
            ModelCheckpoint(dirpath="./checkpoints", monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=7),
            CSVLogger(filepath="./training_log.csv"),
            LRSchedulerCallback(scheduler, monitor="val_loss"),
        ],
    )

    history = trainer.fit(train_loader, val_loader)
    print("\nTraining complete. Summary:")
    print(trainer.metrics_logger.summary())


if __name__ == "__main__":
    main()
