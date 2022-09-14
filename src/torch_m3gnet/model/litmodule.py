from __future__ import annotations

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.linear_model import LinearRegression
from torch.utils.data import random_split
from torch_geometric.data import LightningDataset

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
        self.mae = torchmetrics.MeanAbsoluteError()

    def training_step(self, graph: BatchMaterialGraph, batch_idx: int):
        batch_size = graph[MaterialGraphKey.TOTAL_ENERGY].size(0)
        metrics = self._loss_fn(graph, batch_size)

        # Step scheduler every epoch
        if self.trainer.is_last_batch == 0:
            sch = self.lr_schedulers()
            sch.step()
        return metrics["loss"]

    def validation_step(self, graph: BatchMaterialGraph, batch_idx: int):
        batch_size = graph[MaterialGraphKey.TOTAL_ENERGY].size(0)
        metrics = self._loss_fn(graph, batch_size)

        self.log_dict(
            {f"val_{key}": val for key, val in metrics.items()},
            batch_size=batch_size,
            prog_bar=True,
        )

    def test_step(self, graph: BatchMaterialGraph, batch_idx: int):
        batch_size = graph[MaterialGraphKey.TOTAL_ENERGY].size(0)
        metrics = self._loss_fn(graph, batch_size)

        self.log_dict(
            {f"test_{key}": val for key, val in metrics.items()},
            batch_size=batch_size,
            prog_bar=True,
        )

    def _loss_fn(self, graph: BatchMaterialGraph, batch_size: int):
        num_nodes_per_graph = torch.bincount(
            graph[MaterialGraphKey.BATCH],
            minlength=batch_size,
        )
        target = graph[MaterialGraphKey.TOTAL_ENERGY].clone() / num_nodes_per_graph
        graph = self.model(graph)
        predicted = graph[MaterialGraphKey.TOTAL_ENERGY] / num_nodes_per_graph
        # MSE of energy per atom
        loss = F.mse_loss(predicted, target)

        metrics = {
            "loss": loss,
            "rmse": torch.sqrt(loss),
            "mae": self.mae(predicted, target),
        }
        return metrics

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
    resume_ckpt_path: str | None = None,
    device: str | None = None,
    num_workers: int = -1,
):
    # Device
    if device and device.split(":")[0] == "cuda":
        assert torch.cuda.is_available()
        accelerator = "gpu"
    elif device == "cpu":
        accelerator = "cpu"
    else:
        raise ValueError(f"Unknown or unsupported device: {device}")

    # Fix seed
    seed_everything(config.seed, workers=True)

    # Split dataset
    val_size = int(len(train_and_val) * config.val_ratio)
    train_size = len(train_and_val) - val_size
    train, val = random_split(
        train_and_val,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # Data loader
    if num_workers == -1:
        num_workers = os.cpu_count()  # type: ignore
    datamodule = LightningDataset(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )

    scaled_elemental_energies, mean, std = fit_elemental_energies(
        train.dataset, config.num_types, device
    )

    # Model
    model = build_energy_model(
        cutoff=config.cutoff,
        l_max=config.l_max,
        n_max=config.n_max,
        num_types=config.num_types,
        embedding_dim=config.embedding_dim,
        num_blocks=config.num_blocks,
        scaled_elemental_energies=scaled_elemental_energies,
        mean=mean,
        std=std,
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
        datamodule=datamodule,
        ckpt_path=resume_ckpt_path,
    )

    # Test
    trainer.test(
        model=litmodel,
        datamodule=datamodule,
    )


def fit_elemental_energies(dataset: MaterialGraphDataset, num_types: int, device: torch.device):
    X_all = []
    for graph in dataset:
        one_hot = torch.nn.functional.one_hot(graph[MaterialGraphKey.ATOM_TYPES], num_types)
        X_all.append(torch.sum(one_hot, dim=0).to(torch.float32))
    X_all = torch.stack(X_all).detach().numpy()  # (num_structures, num_types)
    y_all = dataset.data[MaterialGraphKey.TOTAL_ENERGY].numpy()  # (num_structures, )
    mean = np.mean(y_all)
    std = np.std(y_all)
    reg = LinearRegression(fit_intercept=False).fit(X_all, (y_all - mean) / std)
    scaled_elemental_energies = torch.tensor(reg.coef_, device=device)
    return scaled_elemental_energies, mean, std
