"""A simple dense neural network of a desired shape with a single activation
function"""

import torch
import torch.nn as nn


def swish(x):
    return x * nn.functional.sigmoid(x)


class Dense(nn.Module):
    """A model which simply concatenates the parameterization to the inputs"""

    @staticmethod
    def get_layer(in_size, out_size, conv=False):
        if conv:
            return nn.Conv1d(in_size, out_size, 1)
        else:
            return nn.Linear(in_size, out_size)

    def __init__(
        self,
        in_size,
        param_size,
        out_size,
        sizes=None,
        activation=nn.LeakyReLU(0.01),
        final_activation=None,
    ):
        super().__init__()
        if sizes is None:
            sizes = [20, 20, 20, 20, 20]
        self.act = activation
        self.final_act = final_activation
        self.layer0 = nn.Linear(in_size + param_size, sizes[0])
        for i in range(1, len(sizes)):
            setattr(self, f"layer{i}", nn.Linear(sizes[i - 1], sizes[i]))
        setattr(self, f"layer{len(sizes)}", nn.Linear(sizes[-1], out_size))

        self.layers = [
            getattr(self, f"layer{i}") for i in range(len(sizes) + 1)
        ]

    def forward(self, xb, parameterization):
        # Concat the inputs and parameterizations together
        xb = torch.cat((xb, parameterization), dim=1)
        for layer in self.layers[:-1]:
            xb = self.act(layer(xb))
        xb = self.layers[-1](xb)
        if self.final_act is not None:
            xb = self.final_act(xb)
        return xb.view(-1, 1)


class ParameterizedDense(nn.Module):
    """A model which is a standard dense neural network, but after every layer,
    a rescaling is applied element-wise, using a hypernetwork output based on
    the parameterization vector"""

    @staticmethod
    def get_layer(in_size, out_size, conv=False):
        if conv:
            return nn.Conv1d(in_size, out_size, 1)
        else:
            return nn.Linear(in_size, out_size)

    def __init__(
        self,
        in_size,
        param_size,
        out_size,
        sizes=None,
        activation=nn.LeakyReLU(0.01),
        final_activation=None,
    ):
        super().__init__()
        if sizes is None:
            sizes = [20, 20, 20, 20, 20]
        self.sizes = sizes
        self.act = activation
        self.final_act = final_activation
        self.layer0 = nn.Linear(in_size, sizes[0])
        for i in range(1, len(sizes)):
            setattr(self, f"layer{i}", nn.Linear(sizes[i - 1], sizes[i]))
        setattr(self, f"layer{len(sizes)}", nn.Linear(sizes[-1], out_size))

        self.layers = [
            getattr(self, f"layer{i}") for i in range(len(sizes) + 1)
        ]

        self.param_layer = nn.Linear(param_size, sum(sizes))

    def get_parameterized_reweightings(self, parameterization):
        reweightings = self.param_layer(parameterization)
        chunked_reweightings = torch.split(reweightings, self.sizes, dim=-1)
        return chunked_reweightings

    def forward(self, xb, parameterization):
        # Grab the reweightings
        reweightings = self.get_parameterized_reweightings(parameterization)
        for layer, reweighting in zip(self.layers[:-1], reweightings):
            xb = self.act(layer(xb))
            xb = reweighting.view(xb.size()) * xb
        xb = self.layers[-1](xb)
        if self.final_act is not None:
            xb = self.final_act(xb)
        return xb.view(-1, 1)
