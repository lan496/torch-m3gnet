import torch
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.material_graph import BatchMaterialGraph


class DistanceAndAngle(torch.nn.Module):
    """Compute distance between i-j and angle between j-i-k.

    Forward function supplies the following attributes:
        - MaterialGraphKey.EDGE_DISTANCES
        - MaterialGraphKey.TRIPLET_ANGLES

    Note
    ----
    m3gnet.graph._compute.tf_compute_distance_angle
    """

    def forward(self, graph: BatchMaterialGraph) -> BatchMaterialGraph:
        batch = graph[MaterialGraphKey.BATCH]
        lattice = graph[MaterialGraphKey.SCALED_LATTICE]  # unitless
        pos = graph[MaterialGraphKey.SCALED_POS]  # unitless
        edge_index = graph[MaterialGraphKey.EDGE_INDEX]
        edge_cell_shift = graph[MaterialGraphKey.EDGE_CELL_SHIFT]
        triplet_edge_index = graph[MaterialGraphKey.TRIPLET_EDGE_INDEX]

        # Pair distance
        pair_vecs: TensorType["num_edges", 3] = self._get_pair_vectors(batch, lattice, pos, edge_index, edge_cell_shift)  # type: ignore # noqa: F821
        distances: TensorType["num_edges"] = torch.linalg.norm(pair_vecs, dim=1)  # type: ignore # noqa: F821
        vij: TensorType["num_triplets", 3] = pair_vecs[triplet_edge_index[0]]  # type: ignore # noqa: F821

        # Triplet angle
        vik: TensorType["num_triplets", 3] = pair_vecs[triplet_edge_index[1]]  # type: ignore # noqa: F821
        rij: TensorType["num_triplets"] = distances[triplet_edge_index[0]]  # type: ignore # noqa: F821
        rik: TensorType["num_triplets"] = distances[triplet_edge_index[1]]  # type: ignore # noqa: F821
        cos_jik: TensorType["num_triplets"] = torch.sum(vij * vik, axis=1) / (rij * rik)  # type: ignore # noqa: F821

        graph[MaterialGraphKey.EDGE_DISTANCES] = distances
        graph[MaterialGraphKey.TRIPLET_ANGLES] = torch.clamp(cos_jik, min=-1, max=1)

        return graph

    def _get_pair_vectors(
        self,
        batch: TensorType["batch"],  # type: ignore # noqa: F821
        lattice: TensorType["batch", 3, 3],  # type: ignore # noqa: F821
        pos: TensorType["num_nodes", 3],  # type: ignore # noqa: F821
        edge_index: TensorType[2, "num_edges", torch.long],  # type: ignore # noqa: F821
        edge_cell_shift: TensorType["num_edges", 3, torch.int],  # type: ignore # noqa: F821
    ) -> TensorType["num_edges", 3]:  # type: ignore # noqa: F821
        """Return displacements from site-i to site-j."""
        batch_edge: TensorType["num_edges"] = batch[edge_index[0]]  # type: ignore # noqa: F821
        # einsum("ip,ipa->ia", edge_cell_shift, lattice[batch_edge])
        shift_vecs: TensorType["num_edegs", 3] = torch.sum(  # type: ignore # noqa: F821
            edge_cell_shift.to(torch.float)[:, :, None] * lattice[batch_edge], dim=1
        )
        pair_vecs: TensorType["num_edges", 3] = pos[edge_index[1]] + shift_vecs - pos[edge_index[0]]  # type: ignore # noqa: F821
        return pair_vecs
