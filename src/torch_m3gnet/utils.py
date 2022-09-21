from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure


def rotate_cell(structure: Structure, rotation: NDArray) -> Structure:
    lattice = structure.lattice.matrix
    coords = structure.cart_coords

    new_lattice = np.dot(lattice, rotation.T)
    new_coords = np.dot(coords, rotation.T)
    new_frac_coords = np.remainder(np.dot(new_coords, np.linalg.inv(new_lattice)), 1)
    new_coords = np.dot(new_frac_coords, new_lattice)
    return Structure(new_lattice, structure.species, new_coords, coords_are_cartesian=True)


def strain_cell(structure: Structure, strain: NDArray, delta: float) -> Structure:
    lattice = structure.lattice.matrix
    frac_coords = structure.frac_coords
    species = structure.species

    new_lattice = lattice @ (np.eye(3) + delta * strain)
    new_cart_coords = frac_coords @ new_lattice
    new_structure = Structure(new_lattice, species, new_cart_coords, coords_are_cartesian=True)

    return new_structure
