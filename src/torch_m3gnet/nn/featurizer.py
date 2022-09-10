import numpy as np
import torch
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph


class AtomFeaturizer(torch.nn.Module):
    """Featurize atomic numbers.

    Forward function supplies the following attributes:
        - MaterialGraphKey.NODE_FEATURES
    """

    def __init__(self, num_types: int, embedding_dim: int):
        super().__init__()

        self._num_types = num_types

        self.linear = torch.nn.Linear(num_types, embedding_dim, bias=False)

    @property
    def num_types(self) -> int:
        return self._num_types

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        atom_types = graph[MaterialGraphKey.ATOM_TYPES]
        one_hot = torch.nn.functional.one_hot(atom_types, num_classes=self.num_types)
        x = self.linear(one_hot.to(torch.float))
        graph[MaterialGraphKey.NODE_FEATURES] = x
        return graph


class EdgeFeaturizer(torch.nn.Module):
    """Featurize edges by combination of spherical bessel functions.

    Forward function supplies the following attributes:
        - MaterialGraphKey.EDGE_WEIGHTS
        - MaterialGraphKey.EDGE_ATTR
    """

    def __init__(self, degree: int, cutoff: float):
        super().__init__()

        self._degree = degree
        self._cutoff = cutoff

        iota = torch.arange(self.degree)
        self.em = (iota**2) * ((iota + 2) ** 2) / (4 * ((iota + 1) ** 4) + 1)
        dm = torch.ones(self.degree)
        for m in range(1, self.degree):
            dm[m] = 1 - self.em[m] / dm[m - 1]
        self.dm = dm

        self.coeff = torch.empty(self.degree)
        for m in range(self.degree):
            self.coeff[m] = (
                ((-1) ** m)
                * np.sqrt(2)
                * np.pi
                / (self.cutoff**1.5)
                * (m + 1)
                * (m + 2)
                / np.sqrt((m + 1) ** 2 + (m + 2) ** 2)
            )

    @property
    def degree(self) -> int:
        """Return number of basis functions."""
        return self._degree

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        distances: TensorType["num_edges"] = graph[MaterialGraphKey.EDGE_DISTANCES]  # type: ignore # noqa: F821
        num_edges = distances.size(0)

        fm: TensorType["degree", "num_edges"] = torch.empty((self.degree, num_edges))  # type: ignore # noqa: F821
        for m in range(self.degree):
            fm[m] = self.coeff[m] * (
                torch.sinc((m + 1) * torch.pi / self.cutoff * distances)
                + torch.sinc((m + 2) * torch.pi / self.cutoff * distances)
            )

        hm: TensorType["degree", "num_edges"] = torch.empty((self.degree, num_edges))  # type: ignore # noqa: F821
        hm[0] = fm[0]
        for m in range(1, self.degree):
            hm[m] = (fm[m] + torch.sqrt(self.em[m] / self.dm[m - 1]) * hm[m - 1]) / torch.sqrt(
                self.dm[m]
            )

        graph[MaterialGraphKey.EDGE_WEIGHTS]: TensorType["num_edges", "degree"] = torch.transpose(hm, 0, 1)  # type: ignore # noqa: F821
        graph[MaterialGraphKey.EDGE_ATTR]: TensorType["num_edges", "degree"] = graph[MaterialGraphKey.EDGE_WEIGHTS].clone()  # type: ignore # noqa: F821

        return graph
