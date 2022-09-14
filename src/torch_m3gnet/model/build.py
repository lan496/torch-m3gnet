from __future__ import annotations

import torch
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.nn.atom_ref import AtomRef
from torch_m3gnet.nn.conv import M3GNetConv
from torch_m3gnet.nn.featurizer import AtomFeaturizer, EdgeAdjustor, EdgeFeaturizer
from torch_m3gnet.nn.interaction import ThreeBodyInteration
from torch_m3gnet.nn.invariant import DistanceAndAngle
from torch_m3gnet.nn.readout import AtomWiseReadout


def build_energy_model(
    cutoff: float,
    l_max: int,
    n_max: int,
    num_types: int,
    embedding_dim: int,
    num_blocks: int,
    scaled_elemental_energies: TensorType["num_types"] | None = None,  # type: ignore # noqa: F821
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device | None = None,
) -> torch.nn.Sequential:
    degree = n_max * l_max
    num_edge_features = embedding_dim

    if scaled_elemental_energies is None:
        scaled_elemental_energies = torch.zeros(num_types, device=device)

    model = torch.nn.Sequential(
        AtomRef(scaled_elemental_energies, device=device),
        DistanceAndAngle(),
        EdgeFeaturizer(degree=degree, cutoff=cutoff, device=device),
        EdgeAdjustor(degree=degree, num_edge_features=num_edge_features, device=device),
        AtomFeaturizer(num_types=num_types, embedding_dim=embedding_dim, device=device),
    )

    # Convolutions
    for _ in range(num_blocks):
        model.append(
            ThreeBodyInteration(
                cutoff=cutoff,
                l_max=l_max,
                n_max=n_max,
                num_node_features=embedding_dim,
                num_edge_features=num_edge_features,
                device=device,
            )
        )
        model.append(
            M3GNetConv(
                degree=degree,
                num_node_features=embedding_dim,
                num_edge_features=num_edge_features,
                device=device,
            )
        )

    # Readout
    model.append(
        AtomWiseReadout(
            in_features=embedding_dim,
            num_layers=3,
            mean=mean,
            std=std,
            device=device,
        )
    )

    return model
