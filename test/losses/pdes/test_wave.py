import numpy as np
import torch

from pyinsulate.losses.pdes import helmholtz_equation


def test_helmholtz_equation():

    pde = helmholtz_equation

    k = np.random.random()
    phase = np.random.random() * 2 * np.pi
    amplitude = np.random.random() * 100

    size = np.random.randint(1, 100)
    xb = torch.rand(size, 1, requires_grad=True)
    out = torch.sin(k * xb)
    value = pde(out, xb, k=k)

    assert torch.allclose(
        torch.mean(torch.abs(value)), torch.tensor(0.0), atol=1e-7
    )
