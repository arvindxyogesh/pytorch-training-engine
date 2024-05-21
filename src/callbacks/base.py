"""src/callbacks/base.py — Callback base classes."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.trainer import Trainer


class Callback:
    """Base class for all training callbacks."""

    def on_fit_start(self, trainer: "Trainer") -> None: pass
    def on_fit_end(self, trainer: "Trainer") -> None: pass
    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None: pass
    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict) -> None: pass
    def on_batch_start(self, trainer: "Trainer", batch_idx: int) -> None: pass
    def on_batch_end(self, trainer: "Trainer", batch_idx: int, metrics: Dict) -> None: pass


class CallbackList:
    """Orchestrates a list of Callback objects."""

    def __init__(self, callbacks: List[Callback]) -> None:
        self.callbacks = list(callbacks)

    def on_fit_start(self, trainer):
        for cb in self.callbacks: cb.on_fit_start(trainer)

    def on_fit_end(self, trainer):
        for cb in self.callbacks: cb.on_fit_end(trainer)

    def on_epoch_start(self, trainer, epoch):
        for cb in self.callbacks: cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer, epoch, metrics):
        for cb in self.callbacks: cb.on_epoch_end(trainer, epoch, metrics)

    def on_batch_start(self, trainer, batch_idx):
        for cb in self.callbacks: cb.on_batch_start(trainer, batch_idx)

    def on_batch_end(self, trainer, batch_idx, metrics):
        for cb in self.callbacks: cb.on_batch_end(trainer, batch_idx, metrics)
