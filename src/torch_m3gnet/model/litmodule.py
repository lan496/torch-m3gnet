from __future__ import annotations

import os
from dataclasses import asdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.linear_model import LinearRegression
from torch.utils.data import random_split
from torch_geometric.data import LightningDataset

from torch_m3gnet.config import RunConfig
from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.dataset import MaterialGraphDataset
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.model.build import build_model


class LitM3GNet(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        config: RunConfig,
    ):
        super().__init__()
        self.save_hyperparameters(asdict(config))

        self.model = model
        self.config = config

        self.mae = torchmetrics.MeanAbsoluteError()

    def training_step(self, graph: BatchMaterialGraph, batch_idx: int):
        batch_size = graph[MaterialGraphKey.TOTAL_ENERGY].size(0)
        metrics = self._loss_fn(graph, batch_size)

        # https://blog.ceshine.net/post/pytorch-lightning-grad-accu/
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.log_dict(
                {f"train_{key}": val for key, val in metrics.items()},
                batch_size=batch_size,
            )

        return metrics["loss"]

    def training_epoch_end(self, outputs):
        # Step scheduler every epoch
        sch = self.lr_schedulers()
        sch.step()

        return super().training_epoch_end(outputs)

    def validation_step(self, graph: BatchMaterialGraph, batch_idx: int):
        # grad is disabled in validation by default
        # https://github.com/Lightning-AI/lightning/issues/4487
        torch.set_grad_enabled(True)

        batch_size = graph[MaterialGraphKey.TOTAL_ENERGY].size(0)
        metrics = self._loss_fn(graph, batch_size)

        self.log_dict(
            {f"val_{key}": val for key, val in metrics.items()},
            batch_size=batch_size,
        )

    def test_step(self, graph: BatchMaterialGraph, batch_idx: int):
        torch.set_grad_enabled(True)

        batch_size = graph[MaterialGraphKey.TOTAL_ENERGY].size(0)
        metrics = self._loss_fn(graph, batch_size)

        self.log_dict(
            {f"test_{key}": val for key, val in metrics.items()},
            batch_size=batch_size,
        )

    def _loss_fn(self, graph: BatchMaterialGraph, batch_size: int):
        num_nodes_per_graph = torch.bincount(
            graph[MaterialGraphKey.BATCH],
            minlength=batch_size,
        )
        target_energy = graph[MaterialGraphKey.TOTAL_ENERGY].clone() / num_nodes_per_graph
        target_forces = graph[MaterialGraphKey.FORCES].clone()
        target_stresses = graph[MaterialGraphKey.STRESSES].clone()

        # Forward
        graph = self.model(graph)

        predicted_energy = graph[MaterialGraphKey.TOTAL_ENERGY] / num_nodes_per_graph
        predicted_forces = graph[MaterialGraphKey.FORCES]
        predicted_stresses = graph[MaterialGraphKey.STRESSES]

        energy_loss = F.mse_loss(predicted_energy, target_energy, reduction="mean")  # eV/atom
        forces_loss = F.mse_loss(predicted_forces, target_forces, reduction="mean")  # eV/AA
        stresses_loss = F.mse_loss(
            predicted_stresses, target_stresses, reduction="mean"
        )  # eV/AA^3
        loss = (
            energy_loss
            + self.config.force_weight * forces_loss
            + self.config.stress_weight * stresses_loss
        )

        metrics = {
            "loss": loss,
            "energy_loss": energy_loss,
            "forces_loss": forces_loss,
            "stresses_loss": stresses_loss,
            "energy_rmse": torch.sqrt(energy_loss),
            "forces_rmse": torch.sqrt(forces_loss),
            "stresses_rmse": torch.sqrt(stresses_loss),
            "energy_mae": self.mae(predicted_energy, target_energy),
            "forces_mae": self.mae(predicted_forces, target_forces),
            "stresses_mae": self.mae(predicted_stresses, target_stresses),
        }
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.decay_steps,
            eta_min=self.config.learning_rate * self.config.decay_alpha,
        )
        return [optimizer,], [
            scheduler,
        ]

    def on_test_model_eval(self, *args, **kwargs):
        # https://github.com/Lightning-AI/lightning/issues/10287
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)


def train_model(
    train_and_val: MaterialGraphDataset,
    test: MaterialGraphDataset,
    config: RunConfig,
    resume_ckpt_path: str | None = None,
    device: str | None = None,
    num_workers: int = -1,
    debug: bool = False,
):
    accelerator = get_accelerator(device)

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

    scaled_elemental_energies, scale = fit_elemental_energies(
        train.dataset, config.num_types, device
    )

    # Model
    model = build_model(
        cutoff=config.cutoff,
        l_max=config.l_max,
        n_max=config.n_max,
        num_types=config.num_types,
        embedding_dim=config.embedding_dim,
        num_blocks=config.num_blocks,
        scaled_elemental_energies=scaled_elemental_energies,
        scale=scale,
        device=device,
    )
    litmodel = LitM3GNet(
        model=model,
        config=config,
    )

    # Logging
    if debug:
        log_save_dir = f"{config.root}/debug"
    else:
        log_save_dir = f"{config.root}"
    logger = [
        pl_loggers.TensorBoardLogger(save_dir=log_save_dir),
        pl_loggers.CSVLogger(save_dir=log_save_dir),
    ]

    # Trainer
    if debug:
        trainer = pl.Trainer(
            default_root_dir=f"{config.root}/debug",
            callbacks=[
                LearningRateMonitor(logging_interval="epoch"),
                ModelSummary(max_depth=1),
            ],
            max_epochs=config.max_epochs,
            accumulate_grad_batches=config.accumulate_grad_batches,
            logger=logger,
            accelerator=accelerator,
            devices=1,
            overfit_batches=1,
        )
        trainer.fit(
            model=litmodel,
            datamodule=datamodule,
        )
        return

    trainer = pl.Trainer(
        default_root_dir=config.root,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=config.early_stopping_patience)
        ],
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=logger,
        accelerator=accelerator,
        devices=1,
        profiler="simple",
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


def get_accelerator(device: str | torch.device | None) -> str:
    if isinstance(device, torch.device):
        device_str = device.type
    else:
        device_str = device

    if device_str and device_str.split(":")[0] == "cuda":
        assert torch.cuda.is_available()
        accelerator = "gpu"
    elif device_str == "cpu":
        accelerator = "cpu"

    return accelerator


def fit_elemental_energies(
    dataset: MaterialGraphDataset,
    num_types: int,
    device: torch.device,
):
    X_all = []
    for graph in dataset:
        one_hot = torch.nn.functional.one_hot(graph[MaterialGraphKey.ATOM_TYPES], num_types)
        X_all.append(torch.sum(one_hot, dim=0).to(torch.float32))
    X_all = torch.stack(X_all).detach().numpy()  # (num_structures, num_types)
    y_all = dataset.data[MaterialGraphKey.TOTAL_ENERGY].numpy()  # (num_structures, )
    reg = LinearRegression(fit_intercept=False).fit(X_all, y_all)
    elemental_energies = torch.tensor(reg.coef_, device=device)  # eV/atom

    num_atoms = np.sum(X_all, axis=1)
    y_pred = reg.predict(X_all)
    scale = np.sqrt(np.mean(((y_pred - y_all) / num_atoms) ** 2))  # eV/atom
    scaled_elemental_energies = elemental_energies / scale
    return scaled_elemental_energies, scale
