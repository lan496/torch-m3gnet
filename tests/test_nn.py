from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Structure
from torch.autograd import gradcheck
from torch_geometric.data import Batch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph, MaterialGraph
from torch_m3gnet.nn.featurizer import AtomFeaturizer, EdgeFeaturizer
from torch_m3gnet.nn.interaction import legendre_cos, spherical_bessel
from torch_m3gnet.nn.invariant import DistanceAndAngle
from torch_m3gnet.utils import rotate_cell


def test_distance_angle(graph: BatchMaterialGraph, datum: list[MaterialGraph]):
    model = DistanceAndAngle()
    graph = model(graph)

    subgraphs = [Batch.from_data_list([data]) for data in datum]
    subgraphs = [model(subgraph) for subgraph in subgraphs]

    # Check distances in batch
    distances = graph[MaterialGraphKey.EDGE_DISTANCES]
    distances2 = torch.concat(
        [subgraph[MaterialGraphKey.EDGE_DISTANCES] for subgraph in subgraphs]
    )
    torch.testing.assert_close(distances, distances2)

    # Check angles in batch
    angles = graph[MaterialGraphKey.TRIPLET_ANGLES]
    angles2 = torch.concat([subgraph[MaterialGraphKey.TRIPLET_ANGLES] for subgraph in subgraphs])
    torch.testing.assert_close(angles, angles2)


def test_invariance(lattice_coords_types):
    lattice, cart_coords, species = lattice_coords_types
    structure = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
    graph = Batch.from_data_list([MaterialGraph.from_structure(structure, 5.0, 4.0)])

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
    graph2 = Batch.from_data_list([MaterialGraph.from_structure(structure2, 5.0, 4.0)])

    # Distances and angles should be invariant as set w.r.t. rotations
    model = DistanceAndAngle()
    graph = model(graph)
    graph2 = model(graph2)

    torch.testing.assert_close(
        torch.sort(graph[MaterialGraphKey.EDGE_DISTANCES])[0],
        torch.sort(graph2[MaterialGraphKey.EDGE_DISTANCES])[0],
    )
    torch.testing.assert_close(
        torch.sort(graph[MaterialGraphKey.TRIPLET_ANGLES])[0],
        torch.sort(graph2[MaterialGraphKey.TRIPLET_ANGLES])[0],
    )


def test_edge_featurizer(graph: BatchMaterialGraph):
    degree = 3
    model = torch.nn.Sequential(
        DistanceAndAngle(),
        EdgeFeaturizer(degree=degree, cutoff=5.0),
    )
    graph = model(graph)
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.EDGE_WEIGHTS]))
    assert graph[MaterialGraphKey.EDGE_WEIGHTS].size(1) == degree


def test_atom_featurizer(graph: BatchMaterialGraph):
    embedding_dim = 64
    model = AtomFeaturizer(num_types=15, embedding_dim=embedding_dim)
    graph = model(graph)
    assert graph[MaterialGraphKey.NODE_FEATURES].size(1) == embedding_dim
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.NODE_FEATURES]))


@pytest.mark.parametrize(
    "order",
    [0, 1, 2, 3],
)
def test_spherical_bessel(order):
    # Numerical grad near zero give doubtful value...
    input = torch.linspace(1e-1, 10, steps=16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(spherical_bessel, (input, order), eps=1e-4)


@pytest.mark.parametrize(
    "order",
    [0, 1, 2, 3],
)
def test_legendre_cos(order):
    # Numerical grad near zero give doubtful value...
    input = torch.linspace(-1, 1, steps=16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(legendre_cos, (input, order), eps=1e-4)
