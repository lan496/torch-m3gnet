import torch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.model.build import build_model


def test_model(graph: BatchMaterialGraph):
    model = build_model()
    graph = model(graph)


def test_three_body_interaction(graph: BatchMaterialGraph):
    """Model should be invariant with order of triplets."""
    model = build_model()
    graph = model(graph)
    edge_features1 = graph[MaterialGraphKey.EDGE_ATTR].clone()

    num_triplets = graph[MaterialGraphKey.TRIPLET_EDGE_INDEX].size(1)
    perm = torch.randperm(num_triplets)
    graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][0] = graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][0][
        perm
    ]
    graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][1] = graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][1][
        perm
    ]
    graph = model(graph)
    edge_features2 = graph[MaterialGraphKey.EDGE_ATTR].clone()

    torch.testing.assert_close(edge_features1, edge_features2)
    assert not torch.any(torch.isnan(edge_features1))
