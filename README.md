# pytorch-training-engine

A reusable, modular PyTorch training engine inspired by PyTorch Lightning. Packages the training loop, callbacks, checkpointing, and metric logging into composable components that can be dropped into any PyTorch project.

## Architecture

```
pytorch-training-engine/
├── src/
│   ├── engine/
│   │   └── trainer.py       # Main Trainer class
│   ├── callbacks/
│   │   ├── base.py          # Callback + CallbackList base classes
│   │   ├── checkpoint.py    # ModelCheckpoint
│   │   ├── early_stopping.py
│   │   └── logger.py        # CSVLogger, LRSchedulerCallback
│   ├── logging/
│   │   └── metrics_logger.py # In-memory history tracker
│   └── utils/
│       └── helpers.py
├── examples/
│   └── mnist_training.py    # End-to-end example
├── tests/
│   └── test_trainer.py
└── config/config.yaml
```

## Quick Start

```bash
pip install -r requirements.txt
python examples/mnist_training.py
```

## Usage

```python
from src.engine.trainer import Trainer
from src.callbacks.checkpoint import ModelCheckpoint
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.logger import CSVLogger

trainer = Trainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
    max_epochs=100,
    grad_clip=1.0,
    callbacks=[
        ModelCheckpoint(dirpath="./ckpts", monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=10),
        CSVLogger("training_log.csv"),
    ],
)

history = trainer.fit(train_loader, val_loader)
```

## Callback System

| Callback | Purpose |
|---|---|
| `ModelCheckpoint` | Save best model to disk |
| `EarlyStopping` | Stop when metric plateaus |
| `CSVLogger` | Write metrics to CSV |
| `LRSchedulerCallback` | Step any LR scheduler |

Implement `src.callbacks.base.Callback` to add custom callbacks:

```python
class MyCallback(Callback):
    def on_epoch_end(self, trainer, epoch, metrics):
        if metrics["val_acc"] > 0.95:
            print("Target accuracy reached!")
```

## Running Tests

```bash
pytest tests/ -v
```
