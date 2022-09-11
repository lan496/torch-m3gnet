import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from torch_m3gnet.config import RunConfig
from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.dataset import MaterialGraphDataset
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.model.build import build_energy_model


class LitM3GNet(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        decay_steps: int,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps

    def training_step(self, graph: BatchMaterialGraph, batch_idx: int):
        target = graph[MaterialGraphKey.TOTAL_ENERGY].clone()
        graph = self.model(graph)
        predicted = graph[MaterialGraphKey.TOTAL_ENERGY]
        loss = F.mse_loss(predicted, target)

        # Step scheduler every epoch
        if self.trainer.is_last_batch == 0:
            sch = self.lr_schedulers()
            sch.step()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.decay_steps)
        return [optimizer,], [
            scheduler,
        ]


def train_model(
    train: MaterialGraphDataset,
    config: RunConfig,
):
    # Data loader
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)

    # TODO: fit elemental_energies

    # Model
    model = build_energy_model(
        cutoff=config.cutoff,
        l_max=config.l_max,
        n_max=config.n_max,
        num_types=config.num_types,
        embedding_dim=config.embedding_dim,
        num_blocks=config.num_blocks,
        elemental_energies=None,
    )
    litmodel = LitM3GNet(
        model=model,
        learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
    )

    # Trainer
    trainer = pl.Trainer()
    trainer.fit(model=litmodel, train_dataloaders=train_loader)
