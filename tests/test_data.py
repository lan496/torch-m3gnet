from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.dataset import MaterialGraphDataset


def test_batch(graph):
    torch.testing.assert_close(
        graph.batch,
        torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long),
    )
    assert graph.num_nodes == 6
    torch.testing.assert_close(graph.pos.shape, torch.Size([6, 3]))

    # FCC 1st NN: 132 = 12 * 11
    # BCC 1st NN: 56 = 8 * 7
    torch.testing.assert_close(
        graph[MaterialGraphKey.NUM_TRIPLET_I],
        torch.tensor([132, 132, 132, 132, 56, 56]),
    )


def test_dataset(tmpdir: Path):
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

    energies = [2.0, -3.0]
    forces = np.zeros((2, 3))
    stresses = np.zeros((2, 6))
    MaterialGraphDataset(tmpdir, structures, energies, forces, stresses, cutoff, threebody_cutoff)
    # Load processed dataset for second time
    MaterialGraphDataset(tmpdir, structures, energies, forces, stresses, cutoff, threebody_cutoff)
