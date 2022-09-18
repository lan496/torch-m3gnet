import warnings

import numpy as np
import pytest
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
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.SCALED_ATOMIC_ENERGIES]))


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


def test_rotation_invariance(model, lattice_coords_types, rotation, device):
    lattice, cart_coords, species = lattice_coords_types
    structure = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
    graph = Batch.from_data_list([MaterialGraph.from_structure(structure, 5.0, 4.0).to(device)])

    graph = model(graph)
    node_features = graph[MaterialGraphKey.NODE_FEATURES]

    assert np.allclose(np.dot(rotation, rotation.T), np.eye(3))
    lattice2, cart_coords2 = rotate_cell(lattice, cart_coords, rotation)
    structure2 = Structure(lattice2, species, cart_coords2, coords_are_cartesian=True)
    graph2 = Batch.from_data_list([MaterialGraph.from_structure(structure2, 5.0, 4.0).to(device)])

    graph2 = model(graph2)
    node_features2 = graph2[MaterialGraphKey.NODE_FEATURES]

    torch.testing.assert_close(node_features, node_features2)


def test_batch_order(model, datum):
    # Perturb
    torch.manual_seed(0)
    for data in datum:
        data[MaterialGraphKey.POS] += 1e-1 * (torch.rand(data[MaterialGraphKey.POS].shape) - 0.5)

    # Forward in batch
    batch_graph = Batch.from_data_list([data.clone() for data in datum])
    batch_graph = model(batch_graph)

    # Forward for each
    graphs = [Batch.from_data_list([data.clone()]) for data in datum]
    for graph in graphs:
        graph = model(graph)

    energies1 = batch_graph[MaterialGraphKey.TOTAL_ENERGY]
    energies2 = torch.cat([graph[MaterialGraphKey.TOTAL_ENERGY] for graph in graphs])
    torch.testing.assert_close(energies1, energies2)


def test_backward(model, graph: BatchMaterialGraph):
    warnings.filterwarnings("ignore")

    with torch.autograd.detect_anomaly():
        graph = model(graph)
        s = torch.sum(graph[MaterialGraphKey.TOTAL_ENERGY])
        s.backward()


def test_forces(model, graph):
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
                atol=1e-3,
                rtol=1e-2,
            )


def strain_cell(structure: Structure, strain, delta) -> Structure:
    lattice = structure.lattice.matrix
    frac_coords = structure.frac_coords
    species = structure.species

    new_lattice = lattice @ (np.eye(3) + delta * strain)
    new_cart_coords = frac_coords @ new_lattice
    new_structure = Structure(new_lattice, species, new_cart_coords, coords_are_cartesian=True)

    return new_structure


@pytest.mark.skip(reason="Too high noise to compare with numerical difference")
def test_stress(model, lattice_coords_types, device):
    np.random.seed(0)

    cutoff = 5.0
    threebody_cutoff = 4.0

    # batch_size=1 for simplicity
    lattice, cart_coords, species = lattice_coords_types
    cart_coords += 0.5 * (np.random.rand(*cart_coords.shape) - 0.5)
    structure = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
    structure = strain_cell(structure, 0.5, np.random.rand(3, 3) - 0.5)
    graph = Batch.from_data_list(
        [MaterialGraph.from_structure(structure, cutoff, threebody_cutoff).to(device)]
    )

    graph = model(graph)
    stresses_actual = graph[MaterialGraphKey.STRESSES].clone()
    volume = torch.abs(torch.linalg.det(graph[MaterialGraphKey.LATTICE][0]))

    strains = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),  # xx
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # yy
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),  # zz
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),  # yz
        np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),  # zx
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),  # xy
    ]
    delta = 1e-2
    for idx, strain in enumerate(strains):
        structure_plus = strain_cell(structure, strain, delta)
        graph_plus = Batch.from_data_list(
            [MaterialGraph.from_structure(structure_plus, cutoff, threebody_cutoff).to(device)]
        )
        graph_plus = model(graph_plus)
        energy_plus = graph_plus[MaterialGraphKey.TOTAL_ENERGY][0]

        structure_minus = strain_cell(structure, strain, -delta)
        graph_minus = Batch.from_data_list(
            [MaterialGraph.from_structure(structure_minus, cutoff, threebody_cutoff).to(device)]
        )
        graph_minus = model(graph_minus)
        energy_minus = graph_minus[MaterialGraphKey.TOTAL_ENERGY][0]

        stress_expect = -(energy_plus - energy_minus) / (2 * delta * volume)
        print(stress_expect, stresses_actual[0, idx])
