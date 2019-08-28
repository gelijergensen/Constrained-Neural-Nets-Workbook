import numpy as np
import torch

from src.spectral import SpectralReconstruction


def wave_equation(amplitude, frequency, phase, xs):
    return amplitude * torch.sin(2 * np.pi * frequency * xs + phase)


def test_SpectralReconstruction():
    # Create a "neural network" with the reconstruction layer
    net = SpectralReconstruction(1)

    # Create a fake sine wave
    # batch_size = np.random.randint(1, 10)
    batch_size = 1
    amplitude = torch.rand(batch_size, 1) * 10
    frequency = torch.rand(batch_size, 1) * 2
    phase = torch.rand(batch_size, 1) * 2 * np.pi
    # an odd number of points allows us to test the middle of the domain
    xs = torch.linspace(0, 1, steps=5001).view(1, -1)
    wave = wave_equation(amplitude, frequency, phase, xs)

    query_points = xs.unsqueeze(-1)  # torch.rand(amplitude.size())

    val = net(wave, query_points)
    print(f"val: {val}")

    print(
        f"wave(query_points): {wave_equation(amplitude, frequency, phase, query_points)}"
    )
    assert torch.allclose(
        val, wave_equation(amplitude, frequency, phase, query_points)
    )

    assert False

