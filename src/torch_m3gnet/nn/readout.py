import torch
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.nn.core import GatedMLP


class AtomWiseReadout(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        num_layers: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_layers = num_layers

        dimensions = [self.in_features] * (self.num_layers - 1) + [1]
        self.gated = GatedMLP(
            in_features=in_features,
            dimensions=dimensions,
        )

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        x = graph[MaterialGraphKey.NODE_FEATURES]
        atomic_energy: TensorType["num_nodes"] = self.gated(x)  # type: ignore # noqa: F821

        # Elemental energies from AtomRef
        elemental_energies: TensorType["num_nodes"] = graph[MaterialGraphKey.ELEMENTAL_ENERGIES]  # type: ignore # noqa: F821

        graph[MaterialGraphKey.ATOMIC_ENERGIES] = atomic_energy + elemental_energies
        graph[MaterialGraphKey.TOTAL_ENERGY] = torch.sum(graph[MaterialGraphKey.ATOMIC_ENERGIES])
        return graph
