import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from ...load_test_data import get_mnist_dataloaders

from pyinsulate.utils.callbacks import EpochLogger
from pyinsulate.utils.train_model import train_model


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def test_epoch_logger():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    def should_stop(context):
        epoch = context.get('epoch')
        return epoch == 1

    epoch_logger = EpochLogger()

    train_model(model, train_dl, valid_dl, loss_fn, opt,
                should_stop, callbacks=[epoch_logger])

    assert(epoch_logger.epoch == 0)
