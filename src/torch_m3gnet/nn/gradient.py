from __future__ import annotations

import torch
from torch_scatter import scatter_sum
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph


class Gradient(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        energy_scale: float,
        length_scale: float,
    ):
        super().__init__()
        self.model = model
        # self.energy_scale = energy_scale
        # self.length_scale = length_scale

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        # TODO: current implementation cannot be used with spatial decomposition
        graph[MaterialGraphKey.POS].requires_grad_(True)

        graph = self.model(graph)
        grads = torch.autograd.grad(
            torch.sum(graph[MaterialGraphKey.TOTAL_ENERGY]),
            graph[MaterialGraphKey.POS],
            create_graph=True,  # Need to set True for training
        )
        graph[MaterialGraphKey.FORCES] = -grads[0]

        graph[MaterialGraphKey.POS].requires_grad_(False)

        # Virial stress tensor
        num_structures = graph[MaterialGraphKey.SCALED_TOTAL_ENERGY].size(0)
        stresses: TensorType["num_structures", 3, 3] = scatter_sum(  # type: ignore # noqa: F821
            graph[MaterialGraphKey.POS][:, :, None] * graph[MaterialGraphKey.FORCES][:, None, :],
            index=graph[MaterialGraphKey.BATCH],
            dim=0,
            dim_size=num_structures,
        )
        cells = graph[MaterialGraphKey.LATTICE]
        # Volume by scalar triple product, a.(bxc)
        volumes = torch.abs(torch.sum(cells[:, 0] * torch.cross(cells[:, 1], cells[:, 2]), dim=1))
        graph[MaterialGraphKey.STRESSES] = torch.vstack(
            [
                stresses[:, 0, 0],
                stresses[:, 1, 1],
                stresses[:, 2, 2],
                stresses[:, 1, 2],
                stresses[:, 2, 0],
                stresses[:, 0, 1],
            ]
        )
        graph[MaterialGraphKey.STRESSES] = torch.transpose(
            graph[MaterialGraphKey.STRESSES] / volumes, 0, 1
        )

        return graph
