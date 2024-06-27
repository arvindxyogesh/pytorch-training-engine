"""src/callbacks/logger.py — CSVLogger and ConsoleLogger callbacks."""
from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.callbacks.base import Callback

if TYPE_CHECKING:
    from src.engine.trainer import Trainer

logger = logging.getLogger(__name__)


class CSVLogger(Callback):
    """Append epoch metrics to a CSV file."""

    def __init__(self, filepath: str = "training_log.csv") -> None:
        self.filepath = Path(filepath)
        self._writer = None
        self._file = None
        self._header_written = False

    def on_fit_start(self, trainer: "Trainer") -> None:
        self._file = open(self.filepath, "w", newline="")
        logger.info(f"CSVLogger: writing to {self.filepath}")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict) -> None:
        row = {"epoch": epoch, **metrics}
        if not self._header_written:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
            self._header_written = True
        self._writer.writerow(row)
        self._file.flush()

    def on_fit_end(self, trainer: "Trainer") -> None:
        if self._file:
            self._file.close()
            logger.info(f"CSVLogger: closed {self.filepath}")


class LRSchedulerCallback(Callback):
    """Step a learning-rate scheduler at the end of each epoch."""

    def __init__(self, scheduler, monitor: str = "val_loss") -> None:
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict) -> None:
        if hasattr(self.scheduler, "step"):
            if self.monitor in metrics:
                self.scheduler.step(metrics[self.monitor])
            else:
                self.scheduler.step()
