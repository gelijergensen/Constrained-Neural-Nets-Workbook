import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from ..load_test_data import get_mnist_dataloaders

from pyinsulate.utils.callbacks import EpochLogger
from pyinsulate.utils.train_model import training_loop, train_model


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def test_training_loop():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=128)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.9)

    for xb, yb in train_dl:
        model.train()

        original_parameters = next(x.data.clone().detach()
                                   for x in model.parameters())

        training_loop(model, loss_fn, xb, yb, print, list(), False, opt)

        new_parameters = next(x.data.clone().detach()
                              for x in model.parameters())
        assert(not torch.equal(original_parameters, new_parameters))

        training_loop(model, loss_fn, xb, yb, print, list(), True)

        still_new_parameters = next(x.data.clone().detach()
                                    for x in model.parameters())
        assert(torch.equal(new_parameters, still_new_parameters))
        break


def test_train_model():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    def should_stop(context):
        epoch = context.get('epoch')
        return epoch == 1

    train_model(model, train_dl, valid_dl, loss_fn, opt, should_stop)

    assert(True)  # If we get here, we consider this test passed
