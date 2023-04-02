import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

try:
    from torch_geometric.data import LightningDataset
except ImportError:
    pass

from torch_m3gnet.config import RunConfig
from torch_m3gnet.data.dataset import MaterialGraphDataset
from torch_m3gnet.model.litmodule import LitM3GNet, get_accelerator


@pytest.skip
def test_training(
    model: torch.nn.Module,
    config: RunConfig,
    dataset: MaterialGraphDataset,
    device: torch.device,
):
    warnings.filterwarnings("ignore", module="pytorch_lightning")

    seed_everything(config.seed, workers=True)
    accelerator = get_accelerator(device)

    litmodel = LitM3GNet(config=config)
    datamodule = LightningDataset(
        train_dataset=dataset,
        val_dataset=dataset,  # Never do this except testing!
        test_dataset=dataset,  # Never do this except testing!
        batch_size=config.batch_size,
        num_workers=1,
    )

    trainer = pl.Trainer(
        default_root_dir=config.root,
        max_epochs=256,
        accumulate_grad_batches=config.accumulate_grad_batches,
        accelerator=accelerator,
        devices=1,
        overfit_batches=1,  # Overfit to single batch for testing
        inference_mode=False,  # Not to use torch.no_grad() in inference
    )

    trainer.fit(
        model=litmodel,
        datamodule=datamodule,
    )
    metrics = trainer.test(
        datamodule=datamodule,
        ckpt_path="best",
    )
    assert metrics[0]["test_energy_rmse"] < 1e-4  # TODO: somewhat large error?
    assert metrics[0]["test_forces_rmse"] < 1e-6
