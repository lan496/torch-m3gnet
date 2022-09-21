from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from pymatgen.core import Structure
from ruamel.yaml import YAML  # TODO: use hydra

from torch_m3gnet.config import RunConfig
from torch_m3gnet.data.dataset import MaterialGraphDataset
from torch_m3gnet.model.litmodule import train_model


def load_dataset(
    path: Path,
    output_root: Path,
    cutoff: float,
    threebody_cutoff: float,
) -> MaterialGraphDataset:
    with open(path, "r") as f:
        raw_dataset = json.load(f)

    structures = []
    all_energies = []
    all_forces = []
    all_stresses = []
    for data in raw_dataset:
        structure = Structure.from_dict(data["structure"])
        structures.append(structure)
        # eV (total energy)
        all_energies.append(np.array(data["outputs"]["energy"]))
        # eV/AA, (num_sites, 3)
        all_forces.append(np.array(data["outputs"]["forces"]))
        # eV/atom, Voigt order
        # for Si, https://github.com/materialsvirtuallab/mlearn/issues/64
        # 1eV/AA^3 = 1602.1766208 kbar
        # Voigt order: xx, yy, zz, yz, zx, xy
        # VASP order: xx, yy, zz, xy, yz, zx
        vs = np.array(data["outputs"]["virial_stress"]) / 1602.1766208  # eV/AA^3
        all_stresses.append(vs[[0, 1, 2, 5, 3, 4]])

    dataset = MaterialGraphDataset(
        root=output_root,
        structures=structures,
        energies=all_energies,
        forces=all_forces,
        stresses=all_stresses,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
    )

    return dataset


def load_train_dataset(
    path: Path,
    output_root: Path,
    cutoff: float,
    threebody_cutoff: float,
):
    return load_dataset(path / "training.json", output_root, cutoff, threebody_cutoff)


def load_test_dataset(
    path: Path,
    output_root: Path,
    cutoff: float,
    threebody_cutoff: float,
):
    return load_dataset(path / "test.json", output_root, cutoff, threebody_cutoff)


@click.command()
@click.option("--raw_datadir", required=True, type=str, help="Path for mlearn/data.")
@click.option(
    "--element",
    default="all",
    type=str,
    help="One of Cu, Ge, Li, Mo, Ni, Si, and all. 'all' loads the whole six datasets.",
)
@click.option("--config_path", type=str, help="Path for config json.")
@click.option("--resume_ckpt_path", default=None, type=str)
@click.option("--device", default="cpu", help="cuda or cpu")
@click.option("--num_workers", default=-1, type=int)
@click.option("--debug", is_flag=True)
def main(
    raw_datadir: str,
    element: str,
    config_path: str,
    resume_ckpt_path: str | None,
    device: str,
    num_workers: int,
    debug: bool,
):
    # TODO: use hydra
    with open(config_path, "r") as f:
        yaml = YAML()
        config_dict = yaml.load(f)
    config = RunConfig(**config_dict)

    root = Path(config.root)
    elements = ["Cu", "Ge", "Li", "Mo", "Ni", "Si"]
    if element == "all":
        raise NotImplementedError
    elif element in elements:
        train = load_train_dataset(
            path=Path(raw_datadir) / element,
            output_root=root / "train",
            cutoff=config.cutoff,
            threebody_cutoff=config.threebody_cutoff,
        )
        test = load_test_dataset(  # noqa: F841
            path=Path(raw_datadir) / element,
            output_root=root / "test",
            cutoff=config.cutoff,
            threebody_cutoff=config.threebody_cutoff,
        )
    else:
        raise ValueError(f"Dataset for specified element {element} does not exist.")

    train_model(
        config=config,
        train_and_val=(train, test),
        resume_ckpt_path=resume_ckpt_path,
        device=device,
        num_workers=num_workers,
        debug=debug,
    )


if __name__ == "__main__":
    main()
