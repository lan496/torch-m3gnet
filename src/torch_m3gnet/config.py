from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    root: str
    accelerator: str
    devices: int | list[int]
    # Hyperparameter for featurization
    cutoff: float = 5.0
    threebody_cutoff: float = 4.0
    # Hyperparameter for model
    l_max: int = 3
    n_max: int = 3
    num_types: int = 95
    embedding_dim: int = 64
    num_blocks: int = 3
    # Hyperparameter for training
    max_epochs: int = 1000
    learning_rate: float = 1e-3
    decay_steps: int = 100
    batch_size: int = 32
    val_ratio: float = 0.2
    early_stopping_patience: int = 200
    # Misc
    seed: int = 0
    # Devices
    num_workers: int = -1
