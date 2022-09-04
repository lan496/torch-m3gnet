from __future__ import annotations

from typing import Any

import torch
from pymatgen.core import Structure
from torch_geometric.data import Data
from torchtyping import TensorType  # type: ignore
from typing_extensions import TypeAlias

from torch_m3gnet.data import MaterialGraphKey


class MaterialGraph(Data):
    """
    Args
    ----
    x: (num_nodes, num_node_features) torch.float
        Node features
    pos: (num_nodes, 3) torch.float
        Cartesian coordinates
    atom_types: (num_nodes, ) torch.long
        Atomic numbers
    num_triple_i: (num_nodes, ) torch.int
        Number of triplets containing each node

    edge_index: (2, num_edges) torch.long
    edge_attr: (num_edges, num_edge_features) torch.float
        Edge features
    edge_weights: (num_edges, ) torch.float
        Distance of each edge
    edge_cell_shift: (num_edges, 3) torch.int
        Unit-cell vector from source node to destination node
    num_triplet_ij: (num_edges, )
        Number of triplets containing nodes in each edge

    triplet_edge_index: (2, num_triplets) torch.long
        triplet_edge_index[:, f] = [e1, e2] form the `f`-th triplet.
        edge_index[:, e1] = [i, j], edge_index[:, e2] = [i, k]
        Then, [i, j, k] is nodes of the `f`-th triplet.
    triplet_cos_angles: (num_triplets, ) torch.float
        Cosine of angle between j-i-k

    lattice: (3, 3) torch.float
        Row-wise lattice matrix

    num_nodes: int
    num_edges: int
    num_triplets: int
    num_node_features: int
    num_edge_features: int
    """

    def __init__(
        self,
        pos: TensorType["num_nodes", 3] | None = None,  # type: ignore # noqa: F821
        atom_types: TensorType["num_nodes", torch.long] | None = None,  # type: ignore # noqa: F821
        num_triplet_i: TensorType["num_nodes", torch.int] | None = None,  # type: ignore # noqa: F821
        edge_index: TensorType[2, "num_edges", torch.long] | None = None,  # type: ignore # noqa: F821
        edge_cell_shift: TensorType["num_edges", 3, torch.int] | None = None,  # type: ignore # noqa: F821
        num_triplet_ij: TensorType["num_edges", torch.int] | None = None,  # type: ignore # noqa: F821
        triplet_edge_index: TensorType[2, "num_triplets", torch.long] | None = None,  # type: ignore # noqa: F821
        lattice: TensorType[3, 3] | None = None,
    ):
        num_nodes = pos.size(0) if pos is not None else 0
        num_edges = edge_index.size(1) if edge_index is not None else 0
        num_triplets = triplet_edge_index.size(1) if triplet_edge_index is not None else 0

        # TODO: add "state" key

        super().__init__(
            pos=pos,
            atom_types=atom_types,
            num_triplet_i=num_triplet_i,
            edge_index=edge_index,
            edge_cell_shift=edge_cell_shift,
            num_triplet_ij=num_triplet_ij,
            triplet_edge_index=triplet_edge_index,
            lattice=lattice,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_triplets=num_triplets,
            # Derived properties
            edge_distances=None,
            triplet_angles=None,
            edge_weights=None,
            # Features
            x=None,
            edge_attr=None,
        )

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any):
        if key in [MaterialGraphKey.EDGE_INDEX, MaterialGraphKey.TRIPLET_EDGE_INDEX]:
            return 1
        elif key in [MaterialGraphKey.LATTICE]:
            # graph-level properties and so need a new batch dimension
            return None
        else:
            return 0  # cat along node/edge dimension

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == MaterialGraphKey.BATCH:
            return int(value.max()) + 1
        elif key == MaterialGraphKey.EDGE_INDEX:
            return self.num_nodes
        elif key == MaterialGraphKey.TRIPLET_EDGE_INDEX:
            return self.num_edges
        else:
            return 0

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
    ) -> MaterialGraph:
        lattice = torch.as_tensor(structure.lattice.matrix.copy(), dtype=torch.float)
        pos = torch.as_tensor(structure.cart_coords, dtype=torch.float)

        edge_index, edge_cell_shift, distances = get_all_neighbors_with_cell_shifts(
            structure, cutoff
        )

        # Initialize node features by atomic numbers
        atom_types = torch.as_tensor([site.specie.Z for site in structure], dtype=torch.long)

        if threebody_cutoff > cutoff:
            raise ValueError("Three body cutoff raidus should be smaller than two body.")
        num_nodes = len(structure)
        triplet_edge_index, num_triplet_i, num_triplet_ij = compute_threebody(
            num_nodes, edge_index, distances, threebody_cutoff
        )

        return cls(
            pos=pos,
            atom_types=atom_types,
            num_triplet_i=num_triplet_i,
            edge_index=edge_index,
            edge_cell_shift=edge_cell_shift,
            num_triplet_ij=num_triplet_ij,
            triplet_edge_index=triplet_edge_index,
            lattice=lattice,
        )


def get_all_neighbors_with_cell_shifts(
    structure: Structure,
    cutoff: float,
) -> tuple[  # type: ignore
    TensorType[2, "num_edges"],  # noqa: F821
    TensorType["num_edges", 3],  # noqa: F821
    TensorType["num_edges"],  # noqa: F821
]:
    """Construct full neighbor list from pymatgen's structure."""
    all_neighbors = structure.get_all_neighbors(r=cutoff)
    src_indices = []
    dst_indices = []
    edge_cell_shifts = []
    distances = []
    for i, neigbors_i in enumerate(all_neighbors):
        for neighbor in neigbors_i:
            src_indices.append(i)
            dst_indices.append(neighbor.index)
            edge_cell_shifts.append(neighbor.image)
            distances.append(neighbor.nn_distance)

    edge_index = torch.vstack([torch.LongTensor(src_indices), torch.LongTensor(dst_indices)])
    edge_cell_shift = torch.as_tensor(edge_cell_shifts).to(torch.int)
    distances = torch.as_tensor(distances, dtype=torch.float)

    return edge_index, edge_cell_shift, distances


def compute_threebody(
    num_nodes: int,
    edge_index: TensorType[2, "num_edges", torch.long],  # type: ignore # noqa: F821
    distances: TensorType["num_edges"],  # type: ignore # noqa: F821
    threebody_cutoff: float,
) -> tuple[  # type: ignore
    TensorType[2, "num_triplets", torch.long],  # noqa: F821
    TensorType["num_nodes", torch.int],  # noqa: F821
    TensorType["num_edges", torch.int],  # noqa: F821
]:
    """
    Returns
    -------
    triplet_edge_index: (2, num_triplets) torch.long
        triplet_edge_index[:, f] = [e1, e2] form the `f`-th triplet.
        edge_index[:, e1] = [i, j], edge_index[:, e2] = [i, k]
        Then, [i, j, k] is nodes of the `f`-th triplet.
    num_triplet_i: (num_nodes, ) torch.int
        Number of triplets containing each node
    num_triplet_ij: (num_edges, ) torch.int
        Number of triplets containing nodes in each edge

    Note
    ----
    m3gnet.graph._threebody_indices.pyx::compute_threebody
    """
    num_edges = edge_index.size(1)

    valid_edge_index_mask = distances <= threebody_cutoff
    valid_edge_index_mapping = torch.where(valid_edge_index_mask)[0]
    original_indices = torch.arange(num_edges)[valid_edge_index_mask]
    valid_edge_index = edge_index[:, valid_edge_index_mask]

    degrees: TensorType["num_nodes"] = torch.bincount(
        valid_edge_index[1], minlength=num_nodes
    )  # type: ignore # noqa: F821
    num_triplet_i = degrees * (degrees - 1)

    num_triplet = torch.sum(num_triplet_i)
    num_triplet_ij_ = torch.zeros(valid_edge_index.size(1), dtype=torch.int)
    triplet_edge_index = torch.empty((2, num_triplet), dtype=torch.long)
    offset = 0
    idx = 0
    for i in range(num_nodes):
        for j in range(degrees[i]):
            num_triplet_ij_[offset + j] = degrees[i] - 1
            for k in range(degrees[i]):
                if j == k:
                    continue
                triplet_edge_index[0, idx] = offset + j
                triplet_edge_index[1, idx] = offset + k
                idx += 1
        offset += degrees[i]
    triplet_edge_index = original_indices[triplet_edge_index]

    num_triplet_ij = torch.zeros(num_edges, dtype=torch.int)
    num_triplet_ij[valid_edge_index_mapping] = num_triplet_ij_

    return triplet_edge_index, num_triplet_i, num_triplet_ij


# Stab for Batch(_base_cls=MaterialGraph)
BatchMaterialGraph: TypeAlias = Any
