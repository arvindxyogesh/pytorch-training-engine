"""src/callbacks/early_stopping.py — EarlyStopping callback."""
from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING

from src.callbacks.base import Callback

if TYPE_CHECKING:
    from src.engine.trainer import Trainer

logger = logging.getLogger(__name__)


class EarlyStopping(Callback):
    """Stop training when a metric stops improving.

    Args:
        monitor:    Metric to watch (default: 'val_loss').
        patience:   Epochs with no improvement before stopping.
        min_delta:  Minimum improvement threshold.
        mode:       'min' or 'max'.
        restore_best_weights: Not implemented (use ModelCheckpoint for that).
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._wait = 0
        self._compare = (
            (lambda a, b: a < b - min_delta)
            if mode == "min"
            else (lambda a, b: a > b + min_delta)
        )

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict) -> None:
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        if self._best is None or self._compare(current, self._best):
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                logger.info(
                    f"EarlyStopping: {self.monitor} has not improved for "
                    f"{self.patience} epochs. Stopping."
                )
                trainer.stop()
