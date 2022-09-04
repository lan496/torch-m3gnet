from __future__ import annotations

import torch
from torch_geometric.data import Batch

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph, MaterialGraph
from torch_m3gnet.nn.invariant import DistanceAndAngle


def test_distance_angle(graph: BatchMaterialGraph, datum: list[MaterialGraph]):
    model = DistanceAndAngle()
    graph = model(graph)

    subgraphs = [Batch.from_data_list([data]) for data in datum]
    subgraphs = [model(subgraph) for subgraph in subgraphs]

    # Check distances in batch
    distances = graph[MaterialGraphKey.EDGE_WEIGHTS]
    distances2 = torch.concat([subgraph[MaterialGraphKey.EDGE_WEIGHTS] for subgraph in subgraphs])
    torch.testing.assert_close(distances, distances2)

    # Check angles in batch
    angles = graph[MaterialGraphKey.TRIPLET_ANGLES]
    angles2 = torch.concat([subgraph[MaterialGraphKey.TRIPLET_ANGLES] for subgraph in subgraphs])
    torch.testing.assert_close(angles, angles2)
