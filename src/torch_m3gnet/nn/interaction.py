from __future__ import annotations

import math

import torch
from torch_scatter import scatter_sum
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.nn.core import GatedMLP

SPHERICAL_BESSEL_ZEROS = [
    [
        3.141592653589793,
        6.283185307179586,
        9.42477796076938,
        12.566370614359172,
        15.707963267948966,
        18.84955592153876,
        21.991148575128552,
        25.132741228718345,
        28.274333882308138,
        31.41592653589793,
    ],
    [
        4.4934094579080615,
        7.725251836938652,
        10.904121659429897,
        14.066193912832478,
        17.22075527192976,
        20.37130295928857,
        23.519452498688,
        26.666054258813684,
        29.81159879089397,
        32.956389039823485,
    ],
    [
        5.763459196895549,
        9.09501133047736,
        12.322940970567323,
        15.514603010887704,
        18.68903635536381,
        21.853874222710772,
        25.012803202290623,
        28.167829707994635,
        31.320141707448187,
        34.47048833128398,
    ],
    [
        6.987932000501506,
        10.417118547380369,
        13.698023153250254,
        16.923621285214363,
        20.121806174454683,
        23.304246988940612,
        26.476763664540123,
        29.642604540316814,
        32.803732385197115,
        35.96140580471003,
    ],
    [
        8.18256145257219,
        11.704907154571394,
        15.039664707615517,
        18.301255959540985,
        21.525417733400282,
        24.7275655478358,
        27.91557619942227,
        31.093933214080277,
        34.26539008610258,
        37.431736768202505,
    ],
    [
        9.355812111043603,
        12.966530172775347,
        16.35470963934946,
        19.653152101822194,
        22.904550647902713,
        26.127750137225693,
        29.332562578585478,
        32.524661288579686,
        35.707576953062336,
        38.88363095546402,
    ],
    [
        10.512835408094706,
        14.207392458843449,
        17.6479748701669,
        20.983463068945778,
        24.26276804239802,
        27.50786836490526,
        30.730380731646708,
        33.93710830264185,
        37.132331724860926,
        40.31889250922729,
    ],
    [
        11.657032192516846,
        15.43128921026935,
        18.92299919854715,
        22.295348019131772,
        25.60285595381166,
        28.870373347041646,
        32.111196239683615,
        35.33319418271646,
        38.541364851678715,
        41.739052867129445,
    ],
    [
        12.790781711972269,
        16.64100288151313,
        20.182470764950175,
        23.591274817983976,
        26.927040778819027,
        30.217262709362412,
        33.47680081950046,
        36.714529127245726,
        39.93612781086869,
        43.145425017603515,
    ],
    [
        13.915822610504897,
        17.83864319920622,
        21.428486972116353,
        24.873213923876154,
        28.237134359969108,
        31.550188381832864,
        34.828696537686724,
        38.08247908732664,
        41.317864690243525,
        44.5391446334105,
    ],
]


class ThreeBodyInteration(torch.nn.Module):
    """Renormalize triplets to edge features.

    Forward function updates the following attributes:
        - MaterialGraphKey.EDGE_ATTR
    """

    def __init__(
        self,
        cutoff: float,
        l_max: int,
        n_max: int,
        num_node_features: int,
        num_edge_features: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.degree = self.l_max * self.n_max
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        self.spherical_bessel_zeros = torch.tensor(SPHERICAL_BESSEL_ZEROS, device=device)
        if self.spherical_bessel_zeros.size(0) < self.l_max:
            raise ValueError("Too large l_max is specified.")
        if self.spherical_bessel_zeros.size(1) < self.n_max:
            raise ValueError("Too large n_max is specified.")
        self.spherical_bessel_zeros = self.spherical_bessel_zeros[: self.l_max, : self.n_max]

        self.linear_sigmoid1 = torch.nn.Linear(self.num_node_features, self.degree, device=device)
        self.gated_mlp = GatedMLP(
            in_features=self.degree,
            dimensions=[self.num_edge_features],
            device=device,
        )

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        rij = graph[MaterialGraphKey.EDGE_DISTANCES][graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][0]]
        rik = graph[MaterialGraphKey.EDGE_DISTANCES][graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][1]]
        fc_ij: TensorType["num_triplets"] = cutoff_function(rij, self.cutoff)  # type: ignore # noqa: F821
        fc_ik: TensorType["num_triplets"] = cutoff_function(rik, self.cutoff)  # type: ignore # noqa: F821

        angles: TensorType["num_triplets"] = graph[MaterialGraphKey.TRIPLET_ANGLES]  # type: ignore # noqa: F821
        sph: TensorType["l_max", "num_triplets"] = torch.stack(  # type: ignore # noqa: F821
            [
                math.sqrt((2 * order + 1) / math.pi) * legendre_cos(angles, order)
                for order in range(self.l_max)
            ]
        )

        jnlk: TensorType["l_max", "n_max", "num_triplets"] = torch.stack(  # type: ignore # noqa: F821
            [
                spherical_bessel(
                    self.spherical_bessel_zeros[order][:, None] * rik[None, :] / self.cutoff, order
                )
                for order in range(self.l_max)
            ]
        )

        mid_node_features: TensorType["num_nodes", "degree"] = self.linear_sigmoid1(graph[MaterialGraphKey.NODE_FEATURES])  # type: ignore # noqa: F821
        mid_node_features = torch.sigmoid(mid_node_features)
        mid_node_features_reshaped: TensorType["l_max", "n_max", "num_nodes"] = torch.transpose(mid_node_features, 0, 1).reshape(self.l_max, self.n_max, -1)  # type: ignore # noqa: F821

        # Summation over triplets including edge i-j
        node_index_k: TensorType["num_triplets"] = graph[MaterialGraphKey.EDGE_INDEX][1][graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][1]]  # type: ignore # noqa: F821
        mid_edge_features_tmp: TensorType["l_max", "n_max", "num_triplets"] = jnlk * sph[:, None, :] * fc_ij[None, None, :] * fc_ik[None, None, :] * mid_node_features_reshaped[:, :, node_index_k]  # type: ignore # noqa: F821
        num_edges = graph[MaterialGraphKey.EDGE_DISTANCES].size(0)
        mid_edge_features: TensorType["degree", "num_edges"] = scatter_sum(  # type: ignore # noqa: F821
            # ["degree", "num_triplets"]
            mid_edge_features_tmp.reshape(self.degree, -1),
            graph[MaterialGraphKey.TRIPLET_EDGE_INDEX][0],
            dim_size=num_edges,
        )
        mid_edge_features_t: TensorType["num_edges", "degree"] = torch.transpose(mid_edge_features, 0, 1)  # type: ignore # noqa: F821

        edge_update = self.gated_mlp(mid_edge_features_t)
        graph[MaterialGraphKey.EDGE_ATTR] = graph[MaterialGraphKey.EDGE_ATTR] + edge_update

        return graph


class SphericalBessel(torch.autograd.Function):
    """Spherical Bessel function of the first kind."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, order: int):
        EPS = 1e-8
        assert order >= 0

        # torch.sinc(x) = sin(pi * x) / (pi * x)
        out = []
        out.append(
            torch.where(
                input > EPS,
                torch.sin(input) / input,
                torch.ones_like(input),
            )
        )
        if order >= 1:
            out.append(
                torch.where(
                    input > EPS,
                    (torch.sin(input) / input - torch.cos(input)) / input,
                    input / 3,
                )
            )
            coeff = 3
            for n in range(1, order):
                coeff *= 2 * n + 3
                out.append(
                    torch.where(
                        input > EPS,
                        (2 * n + 1) / input * out[n] - out[n - 1],
                        input / coeff,
                    )
                )
        out = torch.stack(out)

        ctx.order = order
        ctx.save_for_backward(input, out)
        return out[-1]

    @staticmethod
    def backward(ctx, grad_output):
        EPS = 1e-8
        input, out = ctx.saved_tensors
        order = ctx.order

        if order == 0:
            grad = torch.where(
                input > EPS,
                -(torch.sin(input) / input - torch.cos(input)) / input * grad_output,
                torch.zeros_like(grad_output),
            )
        elif order == 1:
            grad = torch.where(
                input > EPS,
                (out[order - 1] - (order + 1) / input * out[order]) * grad_output,
                grad_output / 3,
            )
        else:
            grad = torch.where(
                input > EPS,
                (out[order - 1] - (order + 1) / input * out[order]) * grad_output,
                torch.zeros_like(grad_output),
            )

        return grad, None


class LegendreCosPolynomial(torch.autograd.Function):
    """Legendre polynomial P_{n}(cos x)."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, order: int):
        assert order >= 0

        out = [torch.ones_like(input)]
        if order >= 1:
            out.append(input)

            for n in range(1, order):
                out.append(((2 * n + 1) * input * out[n] - n * out[n - 1]) / (n + 1))

        out = torch.stack(out)

        ctx.order = order
        ctx.save_for_backward(input, out)
        return out[-1]

    @staticmethod
    def backward(ctx, grad_output):
        input, out = ctx.saved_tensors
        order = ctx.order

        grad = [torch.zeros_like(grad_output)]
        for n in range(1, order + 1):
            grad.append((n * out[n - 1] + input * grad[n - 1]) * grad_output)

        return grad[-1], None


spherical_bessel = SphericalBessel.apply
legendre_cos = LegendreCosPolynomial.apply


def cutoff_function(input: TensorType, cutoff: float) -> TensorType:
    ratio = input / cutoff
    return 1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3
