import torch


def test_batch(batch):
    torch.testing.assert_close(
        batch.batch,
        torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long),
    )
    assert batch.num_nodes == 6
    torch.testing.assert_close(batch.pos.shape, torch.Size([6, 3]))
