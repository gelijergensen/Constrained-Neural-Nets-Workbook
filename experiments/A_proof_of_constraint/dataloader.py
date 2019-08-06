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

    parameterization = {
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
    }
    wave_equation = construct_wave_equation(**parameterization)

    # Make the data a "vector", not a scalar
    xs = torch.linspace(-1, 1, num_points).unsqueeze(-1)
    ys = wave_equation(xs)
    return xs, ys, parameterization, construct_wave_equation


def get_singlewave_dataloaders(
    frequency,
    phase=None,
    amplitude=1.0,
    num_points=100000,
    num_training=100,
    batch_size=32,
    sampling="start",
    seed=None,
    return_equation=False,
):
    """Gets the singlewave dataloaders given some configurations

    :param frequency: frequency of the wave
    :param phase: phase of the wave. Defaults to a random phase
    :param amplitude: amplitude of the wave
    :param num_points: number of points to sample
    :param num_Training: number of points which are training points
    :param batch_size: batch size of the returned dataloaders
    :param sampling: method for sampling training datapoints. One of:
        "start" - first fraction of the data is training
        "uniform" - training points sampled at a uniform interval across data
        "random" - training points sampled randomly from all data
    :param seed: seed for generating data. If None, then a random seed will be
        picked and returned in the parameterization dictionary
    :param return_equation: whether to additionally return a functional form of
        the data generator
    :return: train_dl, test_dl, parameterization dictionary, and maybe the 
        equation function
    """
    xs, ys, parameterization, equation = make_singlewave_data(
        frequency, phase, amplitude, num_points
    )

    all_idxs = np.arange(0, len(xs))
    if sampling == "start":
        train_idxs = all_idxs[all_idxs < num_training]
    elif sampling == "uniform":
        train_idxs = np.linspace(
            0, len(xs), endpoint=False, num=num_training, dtype=int
        )
    elif sampling == "random":
        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.uint32).max)
        parameterization.update({"seed": seed})
        # Seed the generator
        np.random.seed(seed)
        train_idxs = np.random.permutation(len(xs))[:num_training]
    else:
        print(
            f"Warning! sampling method {sampling} not recognized! Defaulting to 'start'"
        )
        train_idxs = all_idxs[all_idxs < num_training]
    test_idxs = np.delete(all_idxs, train_idxs)

    # These clone, but it's not really an issue because of how small this is
    train_xs = xs[train_idxs].requires_grad_()
    test_xs = xs[test_idxs].requires_grad_()
    train_ys = ys[train_idxs]
    test_ys = ys[test_idxs]

    train_ds = TensorDataset(train_xs, train_ys)
    test_ds = TensorDataset(test_xs, test_ys)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size)
    if return_equation:
        return train_dl, test_dl, parameterization, equation
    else:
        return train_dl, test_dl, parameterization
