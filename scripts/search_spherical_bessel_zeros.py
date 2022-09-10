from functools import partial

import click
import numpy as np
from scipy.optimize import ridder
from scipy.special import spherical_jn


@click.command()
@click.option("--l_max", default=10, type=int)
@click.option("--n_max", default=10, type=int)
def main(l_max, n_max):
    # Ref: https://dlmf.nist.gov/10.21
    jn_zeros = np.zeros((l_max, n_max + l_max - 1))
    # j0 = sinc(x)
    jn_zeros[0, :] = np.pi * np.arange(1, n_max + l_max)
    for order in range(1, l_max):
        jl = partial(spherical_jn, order)

        for n in range(n_max + l_max - order - 1):
            lb = jn_zeros[order - 1, n]
            ub = jn_zeros[order - 1, n + 1]
            jn_zeros[order, n] = ridder(jl, lb, ub)

    for order in range(l_max):
        print("[" + ", ".join(map(str, jn_zeros[order, :n_max])) + "],")


if __name__ == "__main__":
    main()
