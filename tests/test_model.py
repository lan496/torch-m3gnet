from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.model.build import build_model


def test_model(graph: BatchMaterialGraph):
    model = build_model()
    graph = model(graph)
