import torch

from torch_m3gnet.nn.featurizer import AtomFeaturizer, EdgeFeaturizer
from torch_m3gnet.nn.invariant import DistanceAndAngle


def build_model(
    cutoff: float = 5.0,
    n_max: int = 3,
    l_max: int = 3,
    num_types: int = 95,
    embedding_dim: int = 64,
) -> torch.nn.Sequential:
    degree = n_max * l_max
    model = torch.nn.Sequential(
        DistanceAndAngle(),
        EdgeFeaturizer(degree=degree, cutoff=cutoff),
        AtomFeaturizer(num_types=num_types, embedding_dim=embedding_dim),
    )
    return model
