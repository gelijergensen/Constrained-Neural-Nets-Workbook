"""The data for the proof of constraint experiment is a simple model which
is a wave"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


__all__ = ["get_singlewave_dataloaders"]


def construct_wave_equation(amplitude, frequency, phase):
    def wave_equation(xs):
        return amplitude * torch.sin(frequency * xs + phase)
    return wave_equation


def make_singlewave_data(frequency, phase, amplitude, num_points):

    if phase is None:
        phase = np.random.random() * 2 * np.pi

    parameterization = (amplitude, frequency, phase)
    wave_equation = construct_wave_equation(*parameterization)

    # Make the data a "vector", not a scalar
    xs = torch.linspace(0, 2 * np.pi, num_points).unsqueeze(-1)
    ys = wave_equation(xs)
    return xs, ys, (construct_wave_equation, parameterization)


def get_singlewave_dataloaders(frequency, phase=None, amplitude=1.0, num_points=100000, num_training=100, batch_size=32, sampling="start", return_equation=False):
    xs, ys, equation = make_singlewave_data(
        frequency, phase, amplitude, num_points)

    all_idxs = np.arange(0, len(xs))
    if sampling == "start":
        train_idxs = all_idxs < num_training
    elif sampling == "uniform":
        train_idxs = np.linspace(
            0, len(xs), endpoint=False, num=num_training, dtype=int)
    elif sampling == "random":
        train_idxs = np.random.permutation(len(xs))[:num_training]
    else:
        print(
            f"Warning! sampling method {sampling} not recognized! Defaulting to 'start'")
        train_idxs = all_idxs < num_training

    # These clone, but it's not really an issue because of how small this is
    train_xs = xs[train_idxs].requires_grad_()
    test_xs = xs[np.logical_not(train_idxs)].requires_grad_()
    train_ys = ys[train_idxs]
    test_ys = ys[np.logical_not(train_idxs)]

    train_ds = TensorDataset(train_xs, train_ys)
    test_ds = TensorDataset(test_xs, test_ys)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size)
    if return_equation:
        return train_dl, test_dl, equation
    else:
        return train_dl, test_dl
