import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from ...load_test_data import get_mnist_gan_dataloaders

from pyinsulate.utils.callbacks import OutputLogger
from pyinsulate.utils.train_gan_pair import train_gan_pair


class Mnist_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 784)

    def forward(self, xb):
        return self.lin(xb.view(-1, 1))


class Mnist_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 1)

    def forward(self, xb):
        return torch.sigmoid(self.lin(xb))


def test_output_logger():

    train_dl, valid_dl = get_mnist_gan_dataloaders(batch_size=128)
    gener = Mnist_Generator()
    discr = Mnist_Discriminator()
    gener_opt = optim.SGD(gener.parameters(), lr=0.1)
    discr_opt = optim.SGD(discr.parameters(), lr=0.1)

    def should_stop(context):
        epoch = context.get('epoch')
        return epoch == 1

    gen_output_logger = OutputLogger(
        train=True, valid=True, model_type='gener'
    )

    train_gan_pair(
        gener, discr, train_dl, valid_dl, gener_opt, discr_opt, should_stop,
        callbacks=[gen_output_logger]
    )

    assert(len(gen_output_logger.train_outputs) > 0)
    assert(len(gen_output_logger.valid_outputs) > 0)
    assert(gen_output_logger.train_outputs[0].size(
    ) == gen_output_logger.train_yb.size())
    assert(gen_output_logger.valid_outputs[0].size(
    ) == gen_output_logger.valid_yb.size())
