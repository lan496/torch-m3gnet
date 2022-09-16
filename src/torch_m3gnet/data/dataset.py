from __future__ import annotations

import hashlib
import os
from functools import cached_property  # type: ignore
from pathlib import Path

import torch
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pymatgen.core import Structure
from torch_geometric.data import InMemoryDataset

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import MaterialGraph


class MaterialGraphDataset(InMemoryDataset):
    PROCESSED_DATA = "data.pt"

    def __init__(
        self,
        root: Path | str,
        structures: list[Structure],
        energies: NDArray,  # (num_sites, )
        forces: list[NDArray],  # list of (num_sites, 3)
        stresses: list[NDArray],  # list of (6, )
        cutoff: float,
        threebody_cutoff: float,
        n_jobs: int = -1,
    ):
        # Set these variables before super().__init__
        self._root = str(root)
        self._cutoff = cutoff
        self._threebody_cutoff = threebody_cutoff
        self._n_jobs = n_jobs
        self._all_data = [structures, energies, forces, stresses]

        super().__init__(self._root)

        self.data, self.slices = torch.load(self.processed_file_name)

    @property
    def root(self) -> str:
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def threebody_cutoff(self) -> float:
        return self._threebody_cutoff

    @cached_property
    def processed_dir(self) -> str:
        # Override InMemoryDataset
        params_hash: str = hashlib.sha1(
            str((self.cutoff, self.threebody_cutoff)).encode("utf-8")
        ).hexdigest()
        return os.path.join(self.root, f"processed_dataset_{params_hash[:8]}")

    @property
    def processed_file_name(self) -> str:
        return os.path.join(self.processed_dir, self.PROCESSED_DATA)

    @property
    def processed_file_names(self) -> list[str]:
        return [self.PROCESSED_DATA]

    def process(self):
        structures, energies, forces, stresses = self._all_data
        print(f"Process {len(structures)} structures")
        graphs = Parallel(n_jobs=self._n_jobs, verbose=1)(
            delayed(MaterialGraph.from_structure)(structure, self.cutoff, self.threebody_cutoff)
            for structure in structures
        )

        graphs_with_targets = []
        for graph, energy, force, stress in zip(graphs, energies, forces, stresses):
            graph[MaterialGraphKey.TOTAL_ENERGY] = torch.tensor(energy, dtype=torch.float)
            graph[MaterialGraphKey.FORCES] = torch.tensor(force, dtype=torch.float)
            graph[MaterialGraphKey.STRESSES] = torch.tensor(stress, dtype=torch.float)
            graphs_with_targets.append(graph)

        data, slices = self.collate(graphs_with_targets)
        torch.save((data, slices), self.processed_file_name)
