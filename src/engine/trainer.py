"""src/engine/trainer.py — Main Trainer class."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.callbacks.base import Callback, CallbackList
from src.logging.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Reusable PyTorch training engine.

    Usage::

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            max_epochs=50,
            callbacks=[EarlyStopping(patience=5), ModelCheckpoint("./ckpts")],
        )
        history = trainer.fit(train_loader, val_loader)

    Args:
        model:       nn.Module to train.
        optimizer:   Torch optimizer instance.
        criterion:   Loss function.
        max_epochs:  Maximum number of epochs to train.
        device:      'cpu', 'cuda', or 'mps' (auto-detected if None).
        callbacks:   List of Callback objects.
        grad_clip:   If > 0, clips gradient norm to this value.
        log_every:   Log batch-level metrics every N steps.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        max_epochs: int = 100,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        grad_clip: float = 0.0,
        log_every: int = 50,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.log_every = log_every

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.model.to(self.device)

        self.callbacks = CallbackList(callbacks or [])
        self.metrics_logger = MetricsLogger()
        self._stop_training = False
        self.current_epoch = 0
        self.global_step = 0

    # ─────────────────────────────── fit ─────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train model for max_epochs.

        Returns:
            Training history dict with keys 'train_loss', 'val_loss', etc.
        """
        self.callbacks.on_fit_start(self)

        for epoch in range(1, self.max_epochs + 1):
            if self._stop_training:
                logger.info(f"Early stopping triggered at epoch {epoch - 1}.")
                break

            self.current_epoch = epoch
            self.callbacks.on_epoch_start(self, epoch)

            t0 = time.perf_counter()
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._eval_epoch(val_loader) if val_loader else {}

            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_logger.record(epoch, epoch_metrics)

            elapsed = time.perf_counter() - t0
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items())
            logger.info(f"Epoch {epoch:3d}/{self.max_epochs}  {metrics_str}  ({elapsed:.1f}s)")

            self.callbacks.on_epoch_end(self, epoch, epoch_metrics)

        self.callbacks.on_fit_end(self)
        return self.metrics_logger.history

    # ─────────────────────────── epoch loops ─────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            self.callbacks.on_batch_start(self, batch_idx)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.global_step += 1

            total_loss += loss.item() * X.size(0)
            if logits.dim() > 1 and logits.size(1) > 1:
                preds = logits.argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)

            if batch_idx % self.log_every == 0:
                logger.debug(
                    f"  step={self.global_step}  batch_loss={loss.item():.4f}"
                )
            self.callbacks.on_batch_end(self, batch_idx, {"loss": loss.item()})

        n = len(loader.dataset)
        metrics = {"train_loss": total_loss / n}
        if total > 0:
            metrics["train_acc"] = correct / total
        return metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            if logits.dim() > 1 and logits.size(1) > 1:
                preds = logits.argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)

        n = len(loader.dataset)
        metrics = {"val_loss": total_loss / n}
        if total > 0:
            metrics["val_acc"] = correct / total
        return metrics

    # ────────────────────────── prediction ───────────────────────────────────

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> torch.Tensor:
        """Run inference and return concatenated predictions."""
        self.model.eval()
        preds = []
        for X, _ in loader:
            X = X.to(self.device)
            out = self.model(X)
            preds.append(out.cpu())
        return torch.cat(preds, dim=0)

    def stop(self) -> None:
        """Signal the trainer to stop after the current epoch."""
        self._stop_training = True
