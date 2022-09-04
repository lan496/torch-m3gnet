import torch

from torch_m3gnet.data import MaterialGraphKey


def test_batch(graph):
    torch.testing.assert_close(
        graph.batch,
        torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long),
    )
    assert graph.num_nodes == 6
    torch.testing.assert_close(graph.pos.shape, torch.Size([6, 3]))

    # FCC 1st NN: 132 = 12 * 11
    # BCC 1st NN: 56 = 8 * 7
    torch.testing.assert_close(
        graph[MaterialGraphKey.NUM_TRIPLET_I],
        torch.tensor([132, 132, 132, 132, 56, 56]),
    )
