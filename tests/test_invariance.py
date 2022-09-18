from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph, MaterialGraph
from torch_m3gnet.nn.invariant import DistanceAndAngle
from torch_m3gnet.nn.scale import ScaleLength
from torch_m3gnet.utils import rotate_cell, strain_cell


def test_distance_angle_batch(
    graph: BatchMaterialGraph,
    datum: list[MaterialGraph],
):
    model = torch.nn.Sequential(
        ScaleLength(length_scale=1.0),
        DistanceAndAngle(),
    )
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


def test_distance_invariance(lattice_coords_types, rotation):
    lattice, cart_coords, species = lattice_coords_types
    structure = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
    structure = strain_cell(structure, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), delta=0.1)
    graph = Batch.from_data_list([MaterialGraph.from_structure(structure, 5.0, 4.0)])

    assert np.allclose(np.dot(rotation, rotation.T), np.eye(3))
    structure2 = rotate_cell(structure, rotation)
    graph2 = Batch.from_data_list([MaterialGraph.from_structure(structure2, 5.0, 4.0)])

    # Distances and angles should be invariant as set w.r.t. rotations
    model = torch.nn.Sequential(
        ScaleLength(length_scale=1.0),
        DistanceAndAngle(),
    )
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


def test_angle(graph: BatchMaterialGraph):
    model = torch.nn.Sequential(
        ScaleLength(length_scale=1.0),
        DistanceAndAngle(),
    )
    graph = model(graph)
    assert torch.all(graph[MaterialGraphKey.TRIPLET_ANGLES] <= 1)
    assert torch.all(graph[MaterialGraphKey.TRIPLET_ANGLES] >= -1)

    # Sum of angles in each triplet should be pi
    thetas = torch.arccos(graph[MaterialGraphKey.TRIPLET_ANGLES])
    theta_sum: float = torch.sum(thetas).item() / np.pi
    theta_sum -= np.rint(theta_sum)
    assert np.isclose(theta_sum, 0, atol=1e-4, rtol=1e-4)
