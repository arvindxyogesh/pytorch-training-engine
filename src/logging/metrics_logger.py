"""src/logging/metrics_logger.py — In-memory metrics history tracking."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List


class MetricsLogger:
    """Accumulates per-epoch metrics in memory.

    Access history via .history which returns a dict of
    {metric_name: [value_epoch1, value_epoch2, ...]}.
    """

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = defaultdict(list)

    def record(self, epoch: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._history[k].append(float(v))

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    def best(self, metric: str, mode: str = "min") -> float:
        values = self._history.get(metric, [])
        if not values:
            raise KeyError(f"Metric '{metric}' not found in history.")
        return min(values) if mode == "min" else max(values)

    def last(self, metric: str) -> float:
        return self._history[metric][-1]

    def summary(self) -> str:
        lines = []
        for k, vals in self._history.items():
            lines.append(f"  {k}: min={min(vals):.4f}  max={max(vals):.4f}  last={vals[-1]:.4f}")
        return "\n".join(lines)
