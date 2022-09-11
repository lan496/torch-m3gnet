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
    learning_rate: float = 1e-3
    decay_steps: int = 100
    batch_size: int = 32
