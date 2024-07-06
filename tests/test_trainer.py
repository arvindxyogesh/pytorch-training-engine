"""tests/test_trainer.py — Training engine integration tests."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.trainer import Trainer
from src.callbacks.base import Callback
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.checkpoint import ModelCheckpoint
from src.logging.metrics_logger import MetricsLogger


def make_loaders(n=200, n_classes=4, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, 16)
    y = torch.randint(0, n_classes, (n,))
    split = int(n * 0.8)
    tr = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=32, shuffle=True)
    va = DataLoader(TensorDataset(X[split:], y[split:]),  batch_size=64)
    return tr, va


def make_model(n_classes=4):
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, n_classes),
    )


def test_trainer_runs():
    tr, va = make_loaders()
    model = make_model()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        max_epochs=3,
    )
    history = trainer.fit(tr, va)
    assert "train_loss" in history
    assert len(history["train_loss"]) == 3


def test_history_keys():
    tr, va = make_loaders()
    model = make_model()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=nn.CrossEntropyLoss(),
        max_epochs=2,
    )
    history = trainer.fit(tr, va)
    for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
        assert key in history, f"Missing key: {key}"


def test_early_stopping(tmp_path):
    tr, va = make_loaders()
    model = make_model()
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),  # no learning
        criterion=nn.CrossEntropyLoss(),
        max_epochs=20,
        callbacks=[es],
    )
    history = trainer.fit(tr, va)
    # Should stop well before epoch 20
    assert len(history["train_loss"]) <= 10


def test_checkpoint(tmp_path):
    tr, va = make_loaders()
    model = make_model()
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss")
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=nn.CrossEntropyLoss(),
        max_epochs=3,
        callbacks=[ckpt],
    )
    trainer.fit(tr, va)
    assert (tmp_path / "best_model.pt").exists()


def test_metrics_logger():
    ml = MetricsLogger()
    ml.record(1, {"loss": 0.5, "acc": 0.6})
    ml.record(2, {"loss": 0.4, "acc": 0.7})
    assert ml.best("loss", mode="min") == 0.4
    assert ml.best("acc", mode="max") == 0.7
    assert ml.last("loss") == 0.4


def test_predict():
    tr, va = make_loaders()
    model = make_model()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=nn.CrossEntropyLoss(),
        max_epochs=1,
    )
    trainer.fit(tr)
    preds = trainer.predict(va)
    assert preds.shape[0] == 40   # 20% of 200
    assert preds.shape[1] == 4    # n_classes
