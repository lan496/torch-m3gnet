from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Structure
from torch_geometric.data import Batch

from torch_m3gnet.data.material_graph import MaterialGraph


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
            lattice=r_nn * np.sqrt(3) / 2 * np.eye(3),
            species=["Na", "Na"],
            coords=[
                [0, 0, 0],
                [0.5, 0.5, 0.5],
            ],
        ),
    ]
    cutoff = r_nn * np.sqrt(2) + 1e-4
    threebody_cutoff = r_nn + 1e-4
    return [
        MaterialGraph.from_structure(structure, cutoff=cutoff, threebody_cutoff=threebody_cutoff)
        for structure in structures
    ]


@pytest.fixture
def batch(datum) -> Batch:
    return Batch.from_data_list(datum)
