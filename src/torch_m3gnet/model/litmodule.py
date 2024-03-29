from __future__ import annotations

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.linear_model import LinearRegression
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum
from torchtyping import TensorType  # type: ignore

from torch_m3gnet.config import RunConfig
from torch_m3gnet.data import MaterialGraphKey
from torch_m3gnet.data.dataset import MaterialGraphDataset
from torch_m3gnet.data.material_graph import BatchMaterialGraph
from torch_m3gnet.model.build import build_model


class LitM3GNet(pl.LightningModule):
    def __init__(
        self,
        config: RunConfig,
        elemental_energies: list[float] | None = None,
        energy_scale: float = 1.0,
        length_scale: float = 1.0,
        device: str | None = None,
    ):
        super().__init__()

        # Store hyperparameters
        hparams = {}
        hparams["config"] = config
        hparams["elemental_energies"] = elemental_energies  # type: ignore
        hparams["energy_scale"] = energy_scale  # type: ignore
        hparams["length_scale"] = length_scale  # type: ignore
        hparams["device"] = device  # type: ignore
        self.save_hyperparameters(hparams)

        self.config = config

        if elemental_energies:
            elemental_energies_tensor = torch.tensor(elemental_energies, device=device)
        else:
            elemental_energies_tensor = None

        self.model = build_model(
            cutoff=config.cutoff,
            threebody_cutoff=config.threebody_cutoff,
            l_max=config.l_max,
            n_max=config.n_max,
            num_types=config.num_types,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            elemental_energies=elemental_energies_tensor,
            energy_scale=energy_scale,
            length_scale=length_scale,
            device=device,
        )

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def forward(self, graph: BatchMaterialGraph):
        return self.model(graph)

    def training_step(self, graph: BatchMaterialGraph, batch_idx: int):
        batch_size = graph[MaterialGraphKey.TOTAL_ENERGY].size(0)
        metrics = self._loss_fn(graph, batch_size)

        # https://blog.ceshine.net/post/pytorch-lightning-grad-accu/
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.log_dict(
                {f"train_{key}": val for key, val in metrics.items()},
                batch_size=batch_size,
                on_epoch=True,
                on_step=False,
            )

        return metrics["loss"]

    def training_epoch_end(self, outputs):
        # Step scheduler every epoch
        sch = self.lr_schedulers()
        sch.step()

        for name, params in self.named_parameters():
            self.loggers[0].experiment.add_histogram(name, params, self.current_epoch)

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
        num_nodes_per_graph: TensorType["batch_size"] = scatter_sum(  # type: ignore # noqa: F821
            torch.ones_like(graph[MaterialGraphKey.BATCH]),
            index=graph[MaterialGraphKey.BATCH],
            dim_size=batch_size,
        )
        target_energy_per_atom = graph[MaterialGraphKey.TOTAL_ENERGY].clone() / num_nodes_per_graph
        target_forces = graph[MaterialGraphKey.FORCES].clone()
        target_stresses = graph[MaterialGraphKey.STRESSES].clone()

        # Forward
        graph = self(graph)

        predicted_energy_per_atom = graph[MaterialGraphKey.TOTAL_ENERGY] / num_nodes_per_graph
        predicted_forces = graph[MaterialGraphKey.FORCES]
        predicted_stresses = graph[MaterialGraphKey.STRESSES]

        # energy_loss = F.mse_loss(predicted_energy, target_energy, reduction="mean")  # eV
        energy_loss = F.mse_loss(
            predicted_energy_per_atom, target_energy_per_atom, reduction="mean"
        )  # eV/atom
        forces_loss = F.mse_loss(predicted_forces, target_forces, reduction="mean")  # eV/AA
        stresses_loss = F.mse_loss(
            predicted_stresses, target_stresses, reduction="mean"
        )  # eV/AA^3
        loss = (
            self.config.energy_weight * energy_loss
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
            "energy_mae": self.mae(predicted_energy_per_atom, target_energy_per_atom),
            "forces_mae": self.mae(predicted_forces, target_forces),
            "stresses_mae": self.mae(predicted_stresses, target_stresses),
        }
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            eps=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.decay_steps,
            eta_min=self.config.learning_rate * self.config.decay_alpha,
        )
        return [
            optimizer,
        ], [
            scheduler,
        ]

    def on_test_model_eval(self, *args, **kwargs):
        # https://github.com/Lightning-AI/lightning/issues/10287
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        torch.set_grad_enabled(True)
        return self(batch)


def train_model(
    config: RunConfig,
    train_and_val: MaterialGraphDataset | tuple[MaterialGraphDataset, MaterialGraphDataset],
    test: MaterialGraphDataset | None = None,
    resume_ckpt_path: str | None = None,
    device: str | None = None,
    num_workers: int = -1,
    debug: bool = False,
):
    accelerator = get_accelerator(device)

    # Fix seed
    seed_everything(config.seed, workers=True)

    if len(train_and_val) == 2:
        train, val = train_and_val
    else:
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
    # torch_geometric.data.LightningDataset does not seem to care about shuffle flag...
    train_loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val, batch_size=config.batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=False, num_workers=num_workers
    )

    elemental_energies, energy_scale = fit_elemental_energies(train, config.num_types)

    # Model
    litmodel = LitM3GNet(
        config=config,
        elemental_energies=elemental_energies,
        energy_scale=energy_scale,
        # length_scale=length_scale,
        device=device,
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
            ],
            max_epochs=config.max_epochs,
            # https://github.com/Lightning-AI/lightning/discussions/7332
            reload_dataloaders_every_n_epochs=1,  # shuffle train for each epoch
            accumulate_grad_batches=config.accumulate_grad_batches,
            logger=logger,
            accelerator=accelerator,
            devices=1,
            overfit_batches=1,
            log_every_n_steps=10,
        )
        trainer.fit(
            model=litmodel,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        return

    trainer = pl.Trainer(
        default_root_dir=config.root,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=config.early_stopping_patience)
        ],
        max_epochs=config.max_epochs,
        reload_dataloaders_every_n_epochs=1,  # shuffle train for each epoch
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=logger,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        # profiler="simple",
    )
    trainer.fit(
        model=litmodel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt_path,
    )

    # Test
    if test is not None:
        trainer.test(
            model=litmodel,
            dataloaders=test_loader,
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
) -> tuple[list[float], float]:
    X_all = []
    for graph in dataset:
        one_hot = torch.nn.functional.one_hot(graph[MaterialGraphKey.ATOM_TYPES], num_types)
        X_all.append(torch.sum(one_hot, dim=0).to(torch.float32))
    X_all = torch.stack(X_all).detach().numpy()  # (num_structures, num_types)
    y_all = dataset.data[MaterialGraphKey.TOTAL_ENERGY].numpy()  # (num_structures, )
    reg = LinearRegression(fit_intercept=False).fit(X_all, y_all)
    elemental_energies = reg.coef_.tolist()  # eV/atom

    y_pred = reg.predict(X_all)  # eV
    energy_scale = float(np.std(y_pred - y_all))

    # forces = dataset.data[MaterialGraphKey.FORCES]
    # length_scale = energy_scale / torch.std(forces).item()  # AA

    return elemental_energies, energy_scale
