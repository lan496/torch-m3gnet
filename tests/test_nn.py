from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.nn.featurizer import AtomFeaturizer, EdgeFeaturizer
from torch_m3gnet.nn.interaction import legendre_cos, spherical_bessel
from torch_m3gnet.nn.invariant import DistanceAndAngle
from torch_m3gnet.nn.scale import ScaleLength


def test_edge_featurizer(graph: BatchMaterialGraph, device: torch.device):
    degree = 3
    model = torch.nn.Sequential(
        ScaleLength(length_scale=1.0),
        DistanceAndAngle(),
        EdgeFeaturizer(degree=degree, cutoff=5.0, device=device),
    )
    graph = model(graph)
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.EDGE_WEIGHTS]))
    assert graph[MaterialGraphKey.EDGE_WEIGHTS].size(1) == degree


def test_atom_featurizer(graph: BatchMaterialGraph, device: torch.device):
    embedding_dim = 64
    model = AtomFeaturizer(num_types=15, embedding_dim=embedding_dim, device=device)
    graph = model(graph)
    assert graph[MaterialGraphKey.NODE_FEATURES].size(1) == embedding_dim
    assert not torch.any(torch.isnan(graph[MaterialGraphKey.NODE_FEATURES]))


@pytest.mark.parametrize(
    "order",
    [0, 1, 2, 3],
)
def test_spherical_bessel(order):
    # Numerical grad near zero give doubtful value...
    input = torch.linspace(1e-1, 10, steps=16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(spherical_bessel, (input, order), eps=1e-4)


@pytest.mark.parametrize(
    "order",
    [0, 1, 2, 3],
)
def test_legendre_cos(order):
    # Numerical grad near zero give doubtful value...
    input = torch.linspace(-1, 1, steps=16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(legendre_cos, (input, order), eps=1e-4)
