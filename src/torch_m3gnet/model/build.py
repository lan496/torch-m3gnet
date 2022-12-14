from __future__ import annotations

import torch
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.nn.atom_ref import AtomRef
from torch_m3gnet.nn.conv import M3GNetConv
from torch_m3gnet.nn.featurizer import AtomFeaturizer, EdgeAdjustor, EdgeFeaturizer
from torch_m3gnet.nn.gradient import Gradient
from torch_m3gnet.nn.interaction import ThreeBodyInteration
from torch_m3gnet.nn.invariant import DistanceAndAngle
from torch_m3gnet.nn.readout import AtomWiseReadout
from torch_m3gnet.nn.scale import ScaleLength


def build_model(
    cutoff: float,
    threebody_cutoff: float,
    l_max: int,
    n_max: int,
    num_types: int,
    embedding_dim: int,
    num_blocks: int,
    elemental_energies: TensorType["num_types"] | None = None,  # type: ignore # noqa: F821
    energy_scale: float = 1.0,  # eV
    length_scale: float = 1.0,  # AA
    device: torch.device | None = None,
) -> torch.nn.Sequential:
    num_edge_features = embedding_dim

    if elemental_energies is None:
        elemental_energies = torch.zeros(num_types, device=device)

    scaled_cutoff = cutoff / length_scale
    scaled_threebody_cutoff = threebody_cutoff / length_scale

    model = torch.nn.Sequential(
        ScaleLength(length_scale=length_scale),
        AtomRef(elemental_energies, device=device),
        DistanceAndAngle(),
        AtomFeaturizer(num_types=num_types, embedding_dim=embedding_dim, device=device),
        EdgeFeaturizer(degree=n_max, cutoff=scaled_cutoff, device=device),
        EdgeAdjustor(degree=n_max, num_edge_features=num_edge_features, device=device),
    )

    # Convolutions
    for _ in range(num_blocks):
        model.append(
            ThreeBodyInteration(
                cutoff=scaled_cutoff,
                threebody_cutoff=scaled_threebody_cutoff,
                l_max=l_max,
                n_max=n_max,
                num_node_features=embedding_dim,
                num_edge_features=num_edge_features,
                device=device,
            )
        )
        model.append(
            M3GNetConv(
                degree=n_max,
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
            scale=energy_scale,
            device=device,
        )
    )

    # Attach forces
    model = Gradient(
        model,
    )

    return model
