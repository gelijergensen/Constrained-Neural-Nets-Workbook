"""The data for the proof of constraint experiment is a simple model which
is a wave"""

import itertools
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset


__all__ = ["get_singlewave_dataloaders", "get_multiwave_dataloaders"]


def construct_wave_equation(amplitude, frequency, phase):
    def wave_equation(xs):
        return amplitude * torch.sin(2 * np.pi * frequency * xs + phase)

    return wave_equation


class SingleWaveDataset(Dataset):
    @staticmethod
    def make_data(amplitude, frequency, phase, num_points, sampling, seed=None):
        wave_equation = construct_wave_equation(amplitude, frequency, phase)
        if sampling == "random":
            if seed is not None:
                torch.manual_seed(seed)
            xs = 2 * torch.rand((num_points, 1)) - 1
        elif sampling == "uniform":
            xs = torch.linspace(-1, 1, num_points).unsqueeze(-1)
        else:
            raise ValueError(f"Sampling method {sampling} not recognized!")

        ys = wave_equation(xs)
        return xs, ys

    @staticmethod
    def make_parameter_tensor(amplitude, frequency, phase):
        return torch.tensor([amplitude, frequency, phase])

    def __init__(
        self,
        amplitude,
        frequency,
        phase,
        num_points,
        sampling="uniform",
        seed=None,
    ):
        """A dataset with a single example of a wave equation

        :param amplitude: amplitude of the wave
        :param frequency: frequency of the wave
        :param phase: phase of the wave
        :param sampling: method to use for sampling:
            "uniform" - training points sampled with constant interval in [-1,1]
            "random" - training points sampled randomly in [-1,1]
        :param num_points: number of points per parameterization
        :param seed: optional seed for generating data
        """
        self.length = num_points

        self.parameter_tensor = self.make_parameter_tensor(
            amplitude, frequency, phase
        ).requires_grad_()
        self.xy = self.make_data(
            amplitude, frequency, phase, num_points, sampling, seed
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.xy[0][idx].requires_grad_()
        y = self.xy[1][idx]
        param = self.parameter_tensor
        return (x, param), y


class MultiWaveDataset(ConcatDataset):
    def __init__(
        self,
        amplitudes,
        frequencies,
        phases,
        num_points,
        sampling="uniform",
        seed=None,
    ):
        """A dataset with multiple examples of a wave equation

        :param amplitudes: a list of amplitudes
        :param frequencies: a list of frequencies
        :param phases: a list of phases
        :param sampling: method to use for sampling:
            "uniform" - training points sampled with constant interval in [-1,1]
            "random" - training points sampled randomly in [-1,1]
        :param num_points: number of points per parameterization
        :param seed: optional seed for generating data
        """
        parameterizations = [
            {"amplitude": amplitude, "frequency": frequency, "phase": phase}
            for amplitude, frequency, phase in itertools.product(
                amplitudes, frequencies, phases
            )
        ]
        datasets = [
            SingleWaveDataset(
                **parameterization,
                num_points=num_points,
                sampling=sampling,
                seed=seed,
            )
            for parameterization in parameterizations
        ]
        super().__init__(datasets)


def get_singlewave_dataloaders(
    training_parameterization,
    testing_parameterization,
    seed=None,
    batch_size=32,
):
    """Gets the singlewave dataloaders given some configurations

    :param training_parameterization: dictionary with the following keys:
        amplitude: amplitude of wave
        frequencies: frequency of wave
        phases: phase of wave
        sampling: method to use for sampling:
            "uniform" - training points sampled with constant interval in [-1,1]
            "random" - training points sampled randomly in [-1,1]
        num_points: number of points per parameterization
    :param testing_parameterizations: sames as training_parameterizations, but
        for testing data
    :param seed: optional seed for generating data
    :param batch_size: batch size of the returned dataloaders
    :return: train_dl, test_dl
    """
    training_amplitude = training_parameterization["amplitude"]
    training_frequency = training_parameterization["frequency"]
    training_phase = training_parameterization["phase"]
    training_num_points = training_parameterization["num_points"]
    training_sampling = training_parameterization["sampling"]
    train_ds = SingleWaveDataset(
        training_amplitude,
        training_frequency,
        training_phase,
        training_num_points,
        training_sampling,
        seed=seed,
    )
    testing_amplitude = testing_parameterization["amplitude"]
    testing_frequency = testing_parameterization["frequency"]
    testing_phase = testing_parameterization["phase"]
    testing_num_points = testing_parameterization["num_points"]
    testing_sampling = testing_parameterization["sampling"]
    test_ds = SingleWaveDataset(
        testing_amplitude,
        testing_frequency,
        testing_phase,
        testing_num_points,
        testing_sampling,
        seed=seed,
    )
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size)
    return train_dl, test_dl


def get_multiwave_dataloaders(
    training_parameterizations,
    testing_parameterizations,
    seed=None,
    batch_size=32,
):
    """Gets the multiwave dataloaders given some configurations

    :param training_parameterizations: dictionary with the following keys:
        amplitudes: a list of amplitudes
        frequencies: a list of frequencies
        phases: a list of phases
        sampling: method to use for sampling:
            "uniform" - training points sampled with constant interval in [-1,1]
            "random" - training points sampled randomly in [-1,1]
        num_points: number of points per parameterization
    :param testing_parameterizations: sames as training_parameterizations, but
        for testing data
    :param seed: optional seed for generating data
    :param batch_size: batch size of the returned dataloaders
    :return: train_dl, test_dl
    """
    training_amplitudes = training_parameterizations["amplitudes"]
    training_frequencies = training_parameterizations["frequencies"]
    training_phases = training_parameterizations["phases"]
    training_num_points = training_parameterizations["num_points"]
    training_sampling = training_parameterizations["sampling"]
    train_ds = MultiWaveDataset(
        training_amplitudes,
        training_frequencies,
        training_phases,
        training_num_points,
        training_sampling,
        seed=seed,
    )
    testing_amplitudes = testing_parameterizations["amplitudes"]
    testing_frequencies = testing_parameterizations["frequencies"]
    testing_phases = testing_parameterizations["phases"]
    testing_num_points = testing_parameterizations["num_points"]
    testing_sampling = testing_parameterizations["sampling"]
    test_ds = MultiWaveDataset(
        testing_amplitudes,
        testing_frequencies,
        testing_phases,
        testing_num_points,
        testing_sampling,
        seed=seed,
    )
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size)
    return train_dl, test_dl
