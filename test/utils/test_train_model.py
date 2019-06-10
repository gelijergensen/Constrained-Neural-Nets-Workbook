import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from ..load_test_data import get_mnist_dataloaders

from pyinsulate.utils.train_model import batched_loss, train_model


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def test_batched_loss():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=128)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.9)

    for xb, yb in train_dl:
        model.train()

        original_parameters = next(x.data.clone().detach()
                                   for x in model.parameters())

        batched_loss(model, loss_fn, xb, yb, opt)

        new_parameters = next(x.data.clone().detach()
                              for x in model.parameters())
        assert(not torch.equal(original_parameters, new_parameters))

        batched_loss(model, loss_fn, xb, yb)

        still_new_parameters = next(x.data.clone().detach()
                                    for x in model.parameters())
        assert(torch.equal(new_parameters, still_new_parameters))
        break


def test_train_model():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    def until(training_statistics):
        epoch = training_statistics.get('epoch')
        return epoch < 2

    final_epoch = 0

    class Log:
        def __init__(self):
            self.epoch = None

        def __call__(self, training_statistics):
            self.epoch = training_statistics.get('epoch')

    log = Log()

    train_model(model, train_dl, valid_dl, loss_fn, opt, until, log)

    assert(log.epoch == 2)
