from __future__ import annotations

import pickle
from pathlib import Path

import click
import numpy as np
from pymatgen.core import Structure
from ruamel.yaml import YAML  # TODO: use hydra
from tqdm import tqdm

from torch_m3gnet.config import RunConfig
from torch_m3gnet.data.dataset import MaterialGraphDataset
from torch_m3gnet.model.litmodule import train_model


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int):
    np.random.seed(seed)

    val_size = int(n * val_ratio)
    test_size = int(n * test_ratio)
    train_size = n - val_size - test_size
    perm = np.random.permutation(n)
    train_indices = perm[:train_size]
    val_indices = perm[train_size : train_size + val_size]
    test_indicies = perm[train_size + val_size :]

    return train_indices, val_indices, test_indicies


def load_dataset(
    path: Path,
    output_root: Path,
    cutoff: float,
    threebody_cutoff: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    with open(path / "block_0_cif.p", "rb") as f:
        raw_dataset = pickle.load(f)
    with open(path / "block_1_cif.p", "rb") as f:
        raw_dataset.update(pickle.load(f))
    raw_dataset = list(raw_dataset.values())

    # Split train-val-test by material_id
    num_materials = len(raw_dataset)
    train_indices, val_indices, test_indicies = split_indices(
        num_materials, val_ratio, test_ratio, seed
    )

    def _load_dataset_from_indices(indices, save_root: Path):
        if save_root.exists():
            dataset = MaterialGraphDataset(
                root=save_root,
                structures=[],
                energies=[],
                forces=[],
                stresses=[],
                cutoff=cutoff,
                threebody_cutoff=threebody_cutoff,
            )
            return dataset

        structures = []
        all_energies = []
        all_forces = []
        all_stresses = []
        for idx in tqdm(indices):
            data = raw_dataset[idx]

            for cif, energy, forces, stress in zip(
                data["structure"],
                data["energy"],
                data["force"],
                data["stress"],
            ):
                structure = Structure.from_str(cif, fmt="cif")
                structures.append(structure)
                # eV (total energy)
                all_energies.append(energy)
                # eV/AA, (num_sites, 3)
                all_forces.append(np.array(forces))
                # eV/atom, Voigt order
                # 1eV/AA^3 = 1602.1766208 kbar
                vs = np.array(stress) / 1602.1766208  # eV/AA^3
                voigt = np.array([vs[0, 0], vs[1, 1], vs[2, 2], vs[1, 2], vs[2, 0], vs[0, 1]])
                all_stresses.append(voigt)

        dataset = MaterialGraphDataset(
            root=save_root,
            structures=structures,
            energies=all_energies,
            forces=all_forces,
            stresses=all_stresses,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
        )

        return dataset

    train_dataset = _load_dataset_from_indices(train_indices, output_root / "train")
    val_dataset = _load_dataset_from_indices(val_indices, output_root / "val")
    test_dataset = _load_dataset_from_indices(test_indicies, output_root / "test")

    return train_dataset, val_dataset, test_dataset


@click.command()
@click.option("--raw_datadir", required=True, type=str, help="Path for `block_*_cif.p`.")
@click.option("--config_path", type=str, help="Path for config json.")
@click.option("--resume_ckpt_path", default=None, type=str)
@click.option("--device", default="cpu", help="cuda or cpu")
@click.option("--num_workers", default=-1, type=int)
@click.option("--debug", is_flag=True)
def main(
    raw_datadir: str,
    config_path: str,
    resume_ckpt_path: str | None,
    device: str,
    num_workers: int,
    debug: bool,
):
    with open(config_path, "r") as f:
        yaml = YAML()
        config_dict = yaml.load(f)
    config = RunConfig(**config_dict)

    root = Path(config.root)
    train, val, test = load_dataset(
        path=Path(raw_datadir),
        output_root=root,
        cutoff=config.cutoff,
        threebody_cutoff=config.threebody_cutoff,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    train_model(
        config=config,
        train_and_val=(train, val),
        test=test,
        resume_ckpt_path=resume_ckpt_path,
        device=device,
        num_workers=num_workers,
        debug=debug,
    )


if __name__ == "__main__":
    main()
