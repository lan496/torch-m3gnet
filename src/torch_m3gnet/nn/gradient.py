from __future__ import annotations

import torch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph


class Gradient(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
    ):
        super().__init__()
        self.model = model

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        graph[MaterialGraphKey.POS].requires_grad_(True)

        graph = self.model(graph)
        grads = torch.autograd.grad(
            torch.sum(graph[MaterialGraphKey.TOTAL_ENERGY]),
            graph[MaterialGraphKey.POS],
            create_graph=True,  # Need to set True for training
            retain_graph=True,
        )
        graph[MaterialGraphKey.FORCES] = -grads[0]

        graph[MaterialGraphKey.POS].requires_grad_(False)
        return graph
