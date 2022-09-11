import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph, MaterialGraph
from torch_m3gnet.model.build import build_energy_model
from torch_m3gnet.utils import rotate_cell


def test_model(graph: BatchMaterialGraph):
    model = build_energy_model(
        n_max=3,
        l_max=5,
        num_types=93,
        embedding_dim=61,
    )
    graph = model(graph)
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.NODE_FEATURES]))
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.EDGE_ATTR]))
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.ATOMIC_ENERGIES]))


def test_three_body_interaction(graph: BatchMaterialGraph):
    """Model should be invariant with order of triplets."""
    model = build_energy_model()
    graph = model(graph)
    edge_features1 = graph[MaterialGraphKey.EDGE_ATTR].clone()

    num_triplets = graph[MaterialGraphKey.TRIPLET_EDGE_INDEX].size(1)
    perm = torch.randperm(num_triplets)
    graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][0] = graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][0][
        perm
    ]
    graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][1] = graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][1][
        perm
    ]
    graph = model(graph)
    edge_features2 = graph[MaterialGraphKey.EDGE_ATTR].clone()

    torch.testing.assert_close(edge_features1, edge_features2)
    assert not torch.any(torch.isnan(edge_features1))


def test_rotation_invariance(lattice_coords_types):
    model = build_energy_model()

    lattice, cart_coords, species = lattice_coords_types
    structure = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
    graph = Batch.from_data_list([MaterialGraph.from_structure(structure)])

    graph = model(graph)
    node_features = graph[MaterialGraphKey.NODE_FEATURES]

    rotation = np.dot(
        np.array(
            [
                [1 / 2, np.sqrt(3) / 2, 0],
                [-np.sqrt(3) / 2, 1 / 2, 0],
                [0, 0, 1],
            ]
        ),
        np.array(
            [
                [0, 0, 1],
                [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            ]
        ),
    )
    assert np.allclose(np.dot(rotation, rotation.T), np.eye(3))
    lattice2, cart_coords2 = rotate_cell(lattice, cart_coords, rotation)
    structure2 = Structure(lattice2, species, cart_coords2, coords_are_cartesian=True)
    graph2 = Batch.from_data_list([MaterialGraph.from_structure(structure2)])

    graph2 = model(graph2)
    node_features2 = graph2[MaterialGraphKey.NODE_FEATURES]

    torch.testing.assert_close(node_features, node_features2)
