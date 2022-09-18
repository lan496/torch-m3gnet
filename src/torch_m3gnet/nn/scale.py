from __future__ import annotations

import torch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph


class ScaleLength(torch.nn.Module):
    """Normalize length unit

    Forward function supplies the following attributes:
        - MaterialGraphKey.SCALED_POS
        - MaterialGraphKey.SCALED_LATTICE
    """

    def __init__(
        self,
        length_scale: float,
    ):
        super().__init__()
        self.length_scale = length_scale

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        graph[MaterialGraphKey.SCALED_POS] = graph[MaterialGraphKey.POS] / self.length_scale
        graph[MaterialGraphKey.SCALED_LATTICE] = (
            graph[MaterialGraphKey.LATTICE] / self.length_scale
        )
        return graph
