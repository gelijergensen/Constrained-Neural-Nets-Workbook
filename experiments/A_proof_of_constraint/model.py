"""A simple dense neural network of a desired shape with a single activation
function"""

import torch.nn as nn


class Dense(nn.Module):
    """A model which is standard dense neural network, realized either as a
    dense neural network or as a series of 1x1 convolutional filters"""

    @staticmethod
    def get_layer(in_size, out_size, conv=False):
        if conv:
            return nn.Conv1d(in_size, out_size, 1)
        else:
            return nn.Linear(in_size, out_size)

    def __init__(self, in_size, out_size, sizes=None, activation=nn.LeakyReLU(0.01), final_activation=None):
        super().__init__()
        if sizes is None:
            sizes = [20, 20, 20, 20, 20]
        self.act = activation
        self.final_act = final_activation
        self.layer0 = nn.Linear(in_size, sizes[0])
        for i in range(1, len(sizes)):
            setattr(self, f'layer{i}',
                    nn.Linear(sizes[i-1], sizes[i]))
        setattr(self, f'layer{len(sizes)}',
                nn.Linear(sizes[-1], out_size))

        self.layers = [getattr(self, f'layer{i}') for i in range(len(sizes)+1)]

    def forward(self, xb):
        xb = xb.view(-1, 1)
        for layer in self.layers[:-1]:
            xb = self.act(layer(xb))
        xb = self.layers[-1](xb)
        if self.final_act is not None:
            xb = self.final_act(xb)
        return xb.view(-1, 1)
