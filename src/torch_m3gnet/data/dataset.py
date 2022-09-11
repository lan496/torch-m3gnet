from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path

import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import MaterialGraph


class MaterialGraphDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        structures: list[Structure],
        energies: list[float],
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
    ):
        super().__init__()
        self._root = Path(root)
        self._cutoff = cutoff
        self._threebody_cutoff = threebody_cutoff

        if self.processed_file_name.exists():
            self._graphs = torch.load(self.processed_file_name)
            self._processed = False
        else:
            self._graphs = []
            for structure, energy in zip(structures, energies):
                graph = MaterialGraph.from_structure(
                    structure, cutoff=self.cutoff, threebody_cutoff=self.threebody_cutoff
                )
                graph[MaterialGraphKey.TOTAL_ENERGY] = energy
                self._graphs.append(graph)

            self.processed_dir.mkdir()
            torch.save(self._graphs, self.processed_file_name)
            self._processed = True

    @property
    def root(self) -> Path:
        return self._root

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def threebody_cutoff(self) -> float:
        return self._threebody_cutoff

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: str) -> MaterialGraph:
        return self._graphs[idx]

    @cached_property
    def processed_dir(self) -> Path:
        params_hash: str = hashlib.sha1(
            str((self.cutoff, self.threebody_cutoff)).encode("utf-8")
        ).hexdigest()
        return self.root / f"processed_dataset_{params_hash[:8]}"

    @property
    def processed_file_name(self) -> Path:
        return self.processed_dir / "data.pt"
