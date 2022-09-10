import torch
from torch_scatter import scatter_sum
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.nn.core import GatedMLP


class M3GNetConv(torch.nn.Module):
    """Update node and edge features by graph convolution.

    Forward function updates the following attributes:
        - MaterialGraphKey.EDGE_ATTR
        - MaterialGraphKey.NODE_FEATURES
    """

    def __init__(
        self,
        degree: int,
        num_node_features: int,
        num_edge_features: int,
    ):
        super().__init__()
        self.degree = degree
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_concat_features = 2 * num_node_features + num_edge_features

        self.concat_edge_update = GatedMLP(
            in_features=self.num_concat_features,
            dimensions=[self.num_edge_features, self.num_edge_features],
        )
        self.edge_linear = torch.nn.Linear(
            in_features=self.degree, out_features=self.num_edge_features, bias=False
        )

        self.concat_node_update = GatedMLP(
            in_features=self.num_concat_features,
            dimensions=[self.num_edge_features, self.num_node_features],
        )
        self.node_linear = torch.nn.Linear(
            in_features=self.degree, out_features=self.num_node_features, bias=False
        )

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        graph = self.update_edge_features(graph)
        graph = self.update_node_features(graph)
        return graph

    def update_edge_features(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        concat: TensorType["num_edges", "num_concat_features"] = self._concat_features(graph)  # type: ignore # noqa: F821
        graph[MaterialGraphKey.EDGE_ATTR] += self.concat_edge_update(concat) * self.edge_linear(
            graph[MaterialGraphKey.EDGE_WEIGHTS]
        )
        return graph

    def update_node_features(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        concat: TensorType["num_edges", "num_concat_features"] = self._concat_features(graph)  # type: ignore # noqa: F821
        features: TensorType["num_edges", "num_node_features"] = self.concat_node_update(concat) * self.node_linear(graph[MaterialGraphKey.EDGE_WEIGHTS])  # type: ignore # noqa: F821
        num_nodes = graph[MaterialGraphKey.NODE_FEATURES].size(0)
        graph[MaterialGraphKey.NODE_FEATURES] += scatter_sum(
            features,
            index=graph[MaterialGraphKey.EDGE_INDEX][0],
            dim=0,
            dim_size=num_nodes,
        )
        return graph

    @classmethod
    def _concat_features(cls, graph: BatchMaterialGraph):
        vi: TensorType["num_edges", "num_node_features"] = graph[MaterialGraphKey.NODE_FEATURES][graph[MaterialGraphKey.EDGE_INDEX][0]]  # type: ignore # noqa: F821
        vj: TensorType["num_edges", "num_node_features"] = graph[MaterialGraphKey.NODE_FEATURES][graph[MaterialGraphKey.EDGE_INDEX][1]]  # type: ignore # noqa: F821
        eij: TensorType["num_edges", "num_edge_features"] = graph[MaterialGraphKey.EDGE_ATTR]  # type: ignore # noqa: F821
        concat: TensorType["num_edges", "num_concat_features"] = torch.cat([vi, vj, eij], dim=1)  # type: ignore # noqa: F821
        return concat
