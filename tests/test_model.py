import warnings

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph, MaterialGraph
from torch_m3gnet.utils import rotate_cell


def test_model(model, graph: BatchMaterialGraph):
    graph = model(graph)
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.NODE_FEATURES]))
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.EDGE_ATTR]))
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.ATOMIC_ENERGIES]))


def test_three_body_interaction(model, graph: BatchMaterialGraph):
    """Model should be invariant with order of triplets."""
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


def test_rotation_invariance(model, lattice_coords_types, device):
    lattice, cart_coords, species = lattice_coords_types
    structure = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
    graph = Batch.from_data_list([MaterialGraph.from_structure(structure, 5.0, 4.0).to(device)])

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
    graph2 = Batch.from_data_list([MaterialGraph.from_structure(structure2, 5.0, 4.0).to(device)])

    graph2 = model(graph2)
    node_features2 = graph2[MaterialGraphKey.NODE_FEATURES]

    torch.testing.assert_close(node_features, node_features2)


def test_backward(model, graph: BatchMaterialGraph):
    warnings.filterwarnings("ignore")

    with torch.autograd.detect_anomaly():
        graph = model(graph)
        s = torch.sum(graph[MaterialGraphKey.TOTAL_ENERGY])
        s.backward()


def test_gradient(model, graph):
    # Perturb positions
    torch.manual_seed(0)
    graph[MaterialGraphKey.POS] += 1e-1 * (torch.rand(graph[MaterialGraphKey.POS].shape) - 0.5)

    graph = model(graph)
    forces_actual = graph[MaterialGraphKey.FORCES].clone()

    delta = 1e-2
    for node_idx in range(graph[MaterialGraphKey.NUM_NODES]):
        batch_idx = graph[MaterialGraphKey.BATCH][node_idx]
        for direction in range(3):
            graph_plus = graph.clone()
            graph_plus[MaterialGraphKey.POS][node_idx, direction] += delta
            graph_plus = model(graph_plus)
            energy_plus = graph_plus[MaterialGraphKey.TOTAL_ENERGY][batch_idx]

            graph_minus = graph.clone()
            graph_minus[MaterialGraphKey.POS][node_idx, direction] -= delta
            graph_minus = model(graph_minus)
            energy_minus = graph_minus[MaterialGraphKey.TOTAL_ENERGY][batch_idx]

            force_expect = -(energy_plus - energy_minus) / (2 * delta)
            torch.testing.assert_close(
                forces_actual[node_idx, direction],
                force_expect,
                atol=1e-4,
                rtol=1e-4,
            )
