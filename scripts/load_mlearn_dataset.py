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
        # eV/atom
        all_energies.append(np.array(data["outputs"]["energy"]) / data["num_atoms"])
        # eV/AA, (num_sites, 3)
        all_forces.append(np.array(data["outputs"]["forces"]))
        # eV/atom, Voigt order
        # for Si, https://github.com/materialsvirtuallab/mlearn/issues/64
        # 1eV/AA^3 = 1602.1766208 kbar
        # Voigt order: xx, yy, zz, yz, zx, xy
        # VASP order: xx, yy, zz, xy, yz, zx
        vs = np.array(data["outputs"]["virial_stress"]) / structure.num_sites
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
@click.option("--accelerator", default="cpu", type=str)
@click.option("--devices", default=None)
def main(raw_datadir: str, element: str, config_path: str, accelerator: str, devices):
    # TODO: use hydra
    with open(config_path, "r") as f:
        yaml = YAML()
        config_dict = yaml.load(f)
    config = RunConfig(accelerator=accelerator, devices=devices, **config_dict)

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

    train_model(train, test, config)


if __name__ == "__main__":
    main()
