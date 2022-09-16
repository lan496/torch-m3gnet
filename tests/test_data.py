from pathlib import Path

import numpy as np
import torch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.dataset import MaterialGraphDataset


def test_batch(graph):
    torch.testing.assert_close(
        graph.batch.cpu(),
        torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long),
    )
    assert graph.num_nodes == 6
    torch.testing.assert_close(graph.pos.shape, torch.Size([6, 3]))

    # FCC 1st NN: 132 = 12 * 11
    # BCC 1st NN: 56 = 8 * 7
    torch.testing.assert_close(
        graph[MaterialGraphKey.NUM_TRIPLET_I].cpu(),
        torch.tensor([132, 132, 132, 132, 56, 56]),
    )


def test_dataset(tmpdir: Path, structures_and_cutoffs):
    structures, cutoff, threebody_cutoff = structures_and_cutoffs

    energies = np.zeros(len(structures))
    forces = [np.zeros((len(structure), 3)) for structure in structures]
    stresses = [np.zeros(6) for _ in structures]
    MaterialGraphDataset(tmpdir, structures, energies, forces, stresses, cutoff, threebody_cutoff)
    # Load processed dataset for second time
    MaterialGraphDataset(tmpdir, structures, energies, forces, stresses, cutoff, threebody_cutoff)
