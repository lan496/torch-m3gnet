from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    root: str
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
    decay_steps: int = 200
    decay_alpha: float = 1e-2
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    val_ratio: float = 0.2
    early_stopping_patience: int = 200
    energy_weight: float = 1.0
    force_weight: float = 1.0
    stress_weight: float = 0.1
    # Misc
    seed: int = 0
