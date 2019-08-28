import glob
from ignite.engine import Events, create_supervised_trainer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from ..load_test_data import get_mnist_dataloaders, DATA_DIR

from src.handlers import Checkpointer


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


class SimpleCheckpointer(Checkpointer):
    def __init__(self, dirname, filename_base):
        super().__init__(dirname, filename_base, save_interval=1)

        self.constant = "This is a constant"

    def retrieve(self, engine):
        return {"constant": self.constant, "epoch": engine.state.epoch}


def test_checkpointer():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    trainer = create_supervised_trainer(model, opt, loss_fn)

    directory = os.path.join(DATA_DIR, "test_checkpointer")
    checkpointer = SimpleCheckpointer(directory, "simplecheckpointer")
    checkpointer.attach(trainer)

    try:
        num_epochs = 1
        trainer.run(train_dl, max_epochs=num_epochs)
        # load from file and compare
        checkpoint = torch.load(
            os.path.join(
                checkpointer._dirname,
                f"{checkpointer._filename_base}_{checkpointer._iteration:05d}.pth",
            )
        )
        expected = {"constant": checkpointer.constant, "epoch": num_epochs}
        assert checkpoint == expected
    except AssertionError as assertFailed:
        failure = assertFailed
    else:
        failure = None

    # cleanup
    files = glob.glob(f"{directory}/*.pth")
    for f in files:
        os.remove(f)

    if failure is not None:
        raise failure
