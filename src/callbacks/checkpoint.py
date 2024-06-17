"""src/callbacks/checkpoint.py — ModelCheckpoint callback."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import torch

from src.callbacks.base import Callback

if TYPE_CHECKING:
    from src.engine.trainer import Trainer

logger = logging.getLogger(__name__)


class ModelCheckpoint(Callback):
    """Save model weights when a monitored metric improves.

    Args:
        dirpath:   Directory to save checkpoints.
        monitor:   Metric name to monitor (default: 'val_loss').
        mode:      'min' or 'max'.
        save_top_k: Keep only the top-k checkpoints (1 = best only).
        filename:  Checkpoint filename template.
    """

    def __init__(
        self,
        dirpath: str = "./checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        filename: str = "best_model.pt",
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename

        self._best_score: Optional[float] = None
        self._compare = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def on_fit_start(self, trainer: "Trainer") -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict) -> None:
        if self.monitor not in metrics:
            return
        score = metrics[self.monitor]
        if self._best_score is None or self._compare(score, self._best_score):
            self._best_score = score
            save_path = self.dirpath / self.filename
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "metrics": metrics,
                },
                save_path,
            )
            logger.info(
                f"Checkpoint saved: {save_path}  ({self.monitor}={score:.4f})"
            )

    def load_best(self, trainer: "Trainer") -> None:
        path = self.dirpath / self.filename
        if path.exists():
            ckpt = torch.load(path, map_location=trainer.device)
            trainer.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best checkpoint from {path}")
