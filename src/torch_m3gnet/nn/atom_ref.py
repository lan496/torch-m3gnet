import torch
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph


class AtomRef(torch.nn.Module):
    def __init__(
        self,
        elemental_energies: TensorType["num_types"],  # type: ignore # noqa: F821
    ):
        super().__init__()
        self.elemental_energies = elemental_energies

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        atom_types = graph[MaterialGraphKey.ATOM_TYPES]
        energies = self.elemental_energies[atom_types]
        graph[MaterialGraphKey.ELEMENTAL_ENERGIES] = energies
        return graph
