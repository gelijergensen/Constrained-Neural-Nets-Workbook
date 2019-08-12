import numpy as np
import torch

from pyinsulate.pdes import helmholtz_equation, pythagorean_equation


def test_helmholtz_equation():

    pde = helmholtz_equation

    # unbatched
    frequency = torch.rand(1)
    phase = torch.rand(1) * 2 * np.pi
    amplitude = torch.rand(1) * 100
    xb = torch.rand(1, 1, requires_grad=True)
    out = torch.sin(2 * np.pi * frequency * xb + phase)
    value = pde(out, xb, torch.tensor([amplitude, frequency, phase]))

    assert torch.allclose(
        torch.mean(torch.abs(value)), torch.tensor(0.0), atol=1e-5
    )

    # batched
    batch_size = np.random.randint(1, 100)
    frequency = torch.rand(batch_size)
    phase = torch.rand(batch_size) * 2 * np.pi
    amplitude = torch.rand(batch_size) * 100
    xb = torch.rand(batch_size, 1, requires_grad=True)

    out = torch.sin(
        2 * np.pi * frequency.view(xb.size()) * xb + phase.view(xb.size())
    )
    value = pde(out, xb, torch.stack([amplitude, frequency, phase], dim=1))

    assert torch.allclose(
        torch.mean(torch.abs(value)), torch.tensor(0.0), atol=1e-5
    )

    def test_pythagorean_equation():

        pde = pythagorean_equation

        # unbatched
        frequency = torch.rand(1)
        phase = torch.rand(1) * 2 * np.pi
        amplitude = torch.rand(1) * 100
        xb = torch.rand(1, 1, requires_grad=True)
        out = torch.sin(2 * np.pi * frequency * xb + phase)
        value = pde(out, xb, torch.tensor([amplitude, frequency, phase]))

        assert torch.allclose(
            torch.mean(torch.abs(value)), torch.tensor(0.0), atol=1e-8
        )

        # batched
        batch_size = np.random.randint(1, 100)
        frequency = torch.rand(batch_size)
        phase = torch.rand(batch_size) * 2 * np.pi
        amplitude = torch.rand(batch_size) * 100
        xb = torch.rand(batch_size, 1, requires_grad=True)

        out = torch.sin(
            2 * np.pi * frequency.view(xb.size()) * xb + phase.view(xb.size())
        )
        value = pde(out, xb, torch.stack([amplitude, frequency, phase], dim=1))

        assert torch.allclose(
            torch.mean(torch.abs(value)), torch.tensor(0.0), atol=1e-8
        )

