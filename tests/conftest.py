from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from pymatgen.core import Structure
from torch_geometric.data import Batch

from torch_m3gnet.data.material_graph import BatchMaterialGraph, MaterialGraph


@pytest.fixture
def lattice_coords_types() -> tuple[NDArray, NDArray, list[str]]:
    a = 8.01
    lattice = np.eye(3) * a
    coords = np.array(
        [  # in cartesian coords.
            [0.005698, 7.903250, 7.975364],
            [7.962333, 0.031776, 4.087014],
            [7.987993, 4.053572, 7.916418],
            [7.972553, 3.990096, 3.904352],
            [3.901632, 0.009469, 0.015298],
            [4.061435, 7.980741, 3.923483],
            [4.075226, 3.974756, 0.060859],
            [3.997434, 3.997462, 3.900065],
            [0.002131, 2.089909, 2.043724],
            [7.935880, 2.054631, 6.053889],
            [7.986174, 5.996277, 1.901030],
            [0.073084, 5.950515, 5.952990],
            [4.057353, 2.078078, 1.975213],
            [4.049787, 2.018112, 6.084813],
            [3.971569, 5.919147, 2.051521],
            [3.945378, 6.072591, 6.041797],
            [1.964716, 0.069527, 2.062618],
            [1.928378, 7.984901, 6.068134],
            [1.990663, 4.042357, 2.090104],
            [1.974315, 3.921490, 6.056360],
            [6.008068, 7.938413, 2.078371],
            [5.953855, 0.062646, 6.062819],
            [5.900438, 4.009349, 1.999860],
            [6.040758, 3.924354, 6.051151],
            [1.936480, 1.932966, 0.038363],
            [2.043398, 1.921099, 3.956512],
            [1.983471, 5.951049, 0.085619],
            [2.010997, 6.095910, 4.026083],
            [5.955844, 1.984438, 7.911637],
            [6.075395, 1.996245, 4.065586],
            [6.080717, 5.987091, 7.942396],
            [5.983861, 5.933218, 3.927338],
        ]
    )
    types = ["Ti"] * 8 + ["O"] * 24
    return lattice, coords, types


@pytest.fixture
def datum() -> list[MaterialGraph]:
    r_nn = 3.0  # length to the 1st NN
    structures = [
        # Al-fcc
        Structure(
            lattice=r_nn * np.sqrt(2) * np.eye(3),
            species=["Al", "Al", "Al", "Al"],
            coords=[
                [0, 0, 0],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
            ],
        ),
        Structure(
            lattice=r_nn / np.sqrt(3) * 2 * np.eye(3),
            species=["Na", "Na"],
            coords=[
                [0, 0, 0],
                [0.5, 0.5, 0.5],
            ],
        ),
    ]
    cutoff = r_nn + 1e-4
    threebody_cutoff = r_nn + 1e-4
    return [
        MaterialGraph.from_structure(structure, cutoff=cutoff, threebody_cutoff=threebody_cutoff)
        for structure in structures
    ]


@pytest.fixture
def graph(datum) -> BatchMaterialGraph:
    return Batch.from_data_list(datum)
