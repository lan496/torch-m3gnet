from __future__ import annotations

from typing import Any

import torch
from pymatgen.core import Structure
from torch_geometric.data import Data
from torchtyping import TensorType  # type: ignore


class MaterialGraph(Data):
    """
    Args
    ----
    x: (num_nodes, num_node_features) torch.float
        Node features
    pos: (num_nodes, 3) torch.float
        Cartesian coordinates
    num_triple_i: (num_nodes, ) torch.int
        Number of triplets containing each node

    edge_index: (2, num_edges) torch.long
    edge_attr: (num_edges, num_edge_features) torch.float
        Edge features
    edge_weights: (num_edges, ) torch.float
        Distance of each edge
    edge_cell_shift: (num_edges, 3) torch.int
        Unit-cell vector from source node to destination node
    num_triple_ij: (num_edges, )
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
        x: TensorType["num_nodes", "num_node_features"] | None = None,  # type: ignore # noqa: F821
        pos: TensorType["num_nodes", 3] | None = None,  # type: ignore # noqa: F821
        num_triple_i: TensorType["num_nodes", torch.int] | None = None,  # type: ignore # noqa: F821
        edge_index: TensorType[2, "num_edges", torch.long] | None = None,  # type: ignore # noqa: F821
        edge_attr: TensorType["num_edges", "num_edge_features"] | None = None,  # type: ignore # noqa: F821
        edge_weights: TensorType["num_edges"] | None = None,  # type: ignore # noqa: F821
        edge_cell_shift: TensorType["num_edges", 3, torch.int] | None = None,  # type: ignore # noqa: F821
        num_triple_ij: TensorType["num_edges", torch.int] | None = None,  # type: ignore # noqa: F821
        triplet_edge_index: TensorType[2, "num_triplets", torch.long] | None = None,  # type: ignore # noqa: F821
        triplet_cos_angles: TensorType["num_triplets"] | None = None,  # type: ignore # noqa: F821
        lattice: TensorType[3, 3] | None = None,
    ):
        num_nodes = pos.size(0) if pos is not None else 0
        num_edges = edge_index.size(1) if edge_index is not None else 0
        num_triplets = triplet_edge_index.size(1) if triplet_edge_index is not None else 0
        num_node_features = x.size(1) if x is not None else 0
        num_edge_features = edge_attr.size(1) if edge_attr is not None else 0

        # TODO: add "state" key

        super().__init__(
            x=x,
            pos=pos,
            num_triple_i=num_triple_i,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weights=edge_weights,
            edge_cell_shift=edge_cell_shift,
            num_triple_ij=num_triple_ij,
            triplet_edge_index=triplet_edge_index,
            triplet_cos_angles=triplet_cos_angles,
            lattice=lattice,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_triplets=num_triplets,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
        )

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any):
        if key in ["edge_index", "triplet_edge_index"]:
            return 1
        elif key in ["lattice"]:
            # graph-level properties and so need a new batch dimension
            return None
        else:
            return 0  # cat along node/edge dimension

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
        edge_weights = distances

        # Initalize edge features by edge distances
        edge_attr = distances[:, None]
        # Initialize node features by atomic numbers
        x = torch.as_tensor([site.specie.Z for site in structure], dtype=torch.float)[:, None]

        if threebody_cutoff > cutoff:
            raise ValueError("Three body cutoff raidus should be smaller than two body.")
        num_nodes = len(structure)
        triplet_edge_index, num_triple_i, num_triple_ij = compute_threebody(
            num_nodes, edge_index, distances, threebody_cutoff
        )

        pair_vecs = get_pair_vectors(lattice, pos, edge_index, edge_cell_shift)
        triplet_cos_angles = compute_angles(pair_vecs, distances, triplet_edge_index)

        return cls(
            x=x,
            pos=pos,
            num_triple_i=num_triple_i,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weights=edge_weights,
            edge_cell_shift=edge_cell_shift,
            num_triple_ij=num_triple_ij,
            triplet_edge_index=triplet_edge_index,
            triplet_cos_angles=triplet_cos_angles,
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
    num_triple_i: (num_nodes, ) torch.int
        Number of triplets containing each node
    num_triple_ij: (num_edges, ) torch.int
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
    num_triple_i = degrees * (degrees - 1)

    num_triplet = torch.sum(num_triple_i)
    num_triple_ij_ = torch.zeros(valid_edge_index.size(1), dtype=torch.int)
    triplet_edge_index = torch.empty((2, num_triplet), dtype=torch.long)
    offset = 0
    idx = 0
    for i in range(num_nodes):
        for j in range(degrees[i]):
            num_triple_ij_[offset + j] = degrees[i] - 1
            for k in range(degrees[i]):
                if j == k:
                    continue
                triplet_edge_index[0, idx] = offset + j
                triplet_edge_index[1, idx] = offset + k
                idx += 1
        offset += degrees[i]
    triplet_edge_index = original_indices[triplet_edge_index]

    num_triple_ij = torch.zeros(num_edges, dtype=torch.int)
    num_triple_ij[valid_edge_index_mapping] = num_triple_ij_

    return triplet_edge_index, num_triple_i, num_triple_ij


def compute_angles(
    pair_vecs: TensorType["num_edges", 3],  # type: ignore # noqa: F821
    distances: TensorType["num_edges"],  # type: ignore # noqa: F821
    triplet_edge_index: TensorType[2, num_triplets, torch.long],  # type: ignore # noqa: F821
) -> TensorType["num_triplets"]:  # type: ignore # noqa: F821
    """Compute angle between j-i-k."""
    vij: TensorType["num_triplets", 3] = pair_vecs[triplet_edge_index[0]]  # type: ignore # noqa: F821
    vik: TensorType["num_triplets", 3] = pair_vecs[triplet_edge_index[1]]  # type: ignore # noqa: F821
    rij: TensorType["num_triplets"] = distances[triplet_edge_index[0]]  # type: ignore # noqa: F821
    rik: TensorType["num_triplets"] = distances[triplet_edge_index[1]]  # type: ignore # noqa: F821
    cos_jik: TensorType["num_triplets"] = torch.sum(vij * vik, axis=1) / (rij * rik)  # type: ignore # noqa: F821
    return cos_jik


def get_pair_vectors(
    lattice: TensorType[3, 3],
    pos: TensorType["num_nodes", 3],  # type: ignore # noqa: F821
    edge_index: TensorType[2, "num_edges", torch.long],  # type: ignore # noqa: F821
    edge_cell_shift: TensorType["num_edges", 3, torch.int],  # type: ignore # noqa: F821
) -> TensorType["num_edges", 3]:  # type: ignore # noqa: F821
    """Return displacements from site-i to site-j."""
    shift_vecs: TensorType["num_edegs", 3] = (  # type: ignore # noqa: F821
        edge_cell_shift.to(torch.float) @ lattice
    )
    pair_vecs = pos[edge_index[1]] + shift_vecs - pos[edge_index[0]]
    return pair_vecs
