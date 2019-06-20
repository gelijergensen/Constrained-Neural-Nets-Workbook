import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from pyinsulate.utils.callbacks import Checkpointer
from pyinsulate.utils.train_model import train_model

from ...load_test_data import get_mnist_dataloaders


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def test_model_checkpointer():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    def should_stop(context):
        epoch = context.get('epoch')
        return epoch == 2

    checkpointer = Checkpointer(
        ".temp/model_checkpoints", "test_model_checkpointer", save_frequency=1)

    train_model(model, train_dl, valid_dl, loss_fn, opt,
                should_stop, callbacks=[checkpointer])

    for fp in [".temp/model_checkpoints/test_model_checkpointer_1.pkl",
               ".temp/model_checkpoints/test_model_checkpointer_2.pkl"]:
        assert(os.path.isfile(fp))
        with open(fp, "rb") as f:
            loaded_model = torch.load(f)
        assert(len(list(loaded_model.parameters()))
               == len(list(model.parameters())))
        os.remove(fp)
