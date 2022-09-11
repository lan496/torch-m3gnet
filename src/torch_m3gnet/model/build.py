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
    elemental_energies: TensorType["num_types"] | None = None,  # type: ignore # noqa: F821
) -> torch.nn.Sequential:
    degree = n_max * l_max
    num_edge_features = embedding_dim

    if elemental_energies is None:
        elemental_energies = torch.zeros(num_types)

    model = torch.nn.Sequential(
        AtomRef(elemental_energies),
        DistanceAndAngle(),
        EdgeFeaturizer(degree=degree, cutoff=cutoff),
        EdgeAdjustor(degree=degree, num_edge_features=num_edge_features),
        AtomFeaturizer(num_types=num_types, embedding_dim=embedding_dim),
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
            )
        )
        model.append(
            M3GNetConv(
                degree=degree,
                num_node_features=embedding_dim,
                num_edge_features=num_edge_features,
            )
        )

    # Readout
    model.append(
        AtomWiseReadout(
            in_features=embedding_dim,
            num_layers=3,
        )
    )

    return model
