from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotate_cell(lattice: NDArray, coords: NDArray, rotation: NDArray) -> tuple[NDArray, NDArray]:
    new_lattice = np.dot(lattice, rotation.T)
    new_coords = np.dot(coords, rotation.T)
    new_frac_coords = np.remainder(np.dot(new_coords, np.linalg.inv(new_lattice)), 1)
    new_coords = np.dot(new_frac_coords, new_lattice)
    return new_lattice, new_coords
