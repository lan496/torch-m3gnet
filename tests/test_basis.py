from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from torch_m3gnet.nn.interaction import (
    SPHERICAL_BESSEL_ZEROS,
    legendre_cos,
    spherical_bessel,
)


def test_spherical_bessel_zeros():
    l_max = len(SPHERICAL_BESSEL_ZEROS)
    for order in range(l_max):
        for root in SPHERICAL_BESSEL_ZEROS[order]:
            torch.testing.assert_close(
                spherical_bessel(torch.tensor([root]), order),
                torch.tensor([0.0], dtype=torch.float),
            )


@pytest.mark.parametrize(
    "order",
    [0, 1, 2, 3],
)
def test_spherical_bessel(order):
    # Numerical grad near zero give doubtful value...
    input = torch.linspace(1e-1, 10, steps=16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(spherical_bessel, (input, order), eps=1e-4)


@pytest.mark.parametrize(
    "order",
    [0, 1, 2, 3],
)
def test_legendre_cos(order):
    # Numerical grad near zero give doubtful value...
    input = torch.linspace(-1, 1, steps=16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(legendre_cos, (input, order), eps=1e-4)
