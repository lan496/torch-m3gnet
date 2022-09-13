import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import random_split
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
        graph = graph  # to(device)
        import IPython; IPython.embed(colors='neutral')

        target = graph[MaterialGraphKey.TOTAL_ENERGY].clone()
        graph = self.model(graph)
        predicted = graph[MaterialGraphKey.TOTAL_ENERGY]
        loss = F.mse_loss(predicted, target)

        # Step scheduler every epoch
        if self.trainer.is_last_batch == 0:
            sch = self.lr_schedulers()
            sch.step()
        return loss

    def validation_step(self, graph: BatchMaterialGraph, batch_idx: int):
        graph = graph  # to(device)

        target = graph[MaterialGraphKey.TOTAL_ENERGY].clone()
        graph = self.model(graph)
        predicted = graph[MaterialGraphKey.TOTAL_ENERGY]
        loss = F.mse_loss(predicted, target)

        batch_size = target.size(0)
        self.log("val_loss", loss, batch_size=batch_size)

    def test_step(self, graph: BatchMaterialGraph, batch_idx: int):
        graph = graph  # to(device)

        target = graph[MaterialGraphKey.TOTAL_ENERGY].clone()
        graph = self.model(graph)
        predicted = graph[MaterialGraphKey.TOTAL_ENERGY]
        loss = F.mse_loss(predicted, target)

        batch_size = target.size(0)
        self.log("test_loss", loss, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.decay_steps)
        return [optimizer,], [
            scheduler,
        ]


def train_model(
    train_and_val: MaterialGraphDataset,
    test: MaterialGraphDataset,
    config: RunConfig,
    device: str | None = None,
):
    # Device
    if device.split(':')[0] == 'cuda':
        assert torch.cuda.is_available()
        accelerator = 'gpu'
    elif device == 'cpu':
        accelerator = 'cpu'
    else:
        raise ValueError(f"Unknown or unsupported accelerator: {config.accelerator}")

    # Fix seed
    seed = torch.manual_seed(config.seed)

    # Split dataset
    val_size = int(len(train_and_val) * config.val_ratio)
    train_size = len(train_and_val) - val_size
    train, val = random_split(train_and_val, [train_size, val_size], generator=seed)

    # Data loader
    num_workers = config.num_workers
    if num_workers == -1:
        num_workers = os.cpu_count()  # type: ignore
    train_loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val, batch_size=config.batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=config.batch_size, shuffle=False, num_workers=num_workers
    )

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
        device=device,
    )
    litmodel = LitM3GNet(
        model=model,
        learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
    )

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=config.root,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=config.early_stopping_patience)
        ],
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,
    )
    trainer.fit(
        model=litmodel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Test
    trainer.test(model, dataloaders=test_loader)
