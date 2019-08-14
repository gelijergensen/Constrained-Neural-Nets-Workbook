from ignite.engine import Events, create_supervised_trainer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from ..load_test_data import get_mnist_dataloaders

from pyinsulate.handlers import Monitor


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


class DummyMonitor(Monitor):
    def __init__(self):
        super().__init__()
        # epoch is now automatically retrieved

        self.add_key("dummy", mean=True, std=True, percentiles=[0.0, 100.0])

    def __call__(self, engine):
        self.add_value("dummy", self._iterations_per_epoch[-1])


def test_monitor():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    trainer = create_supervised_trainer(model, opt, loss_fn)

    monitor = DummyMonitor()
    monitor.attach(trainer)

    num_epochs = 1
    trainer.run(train_dl, max_epochs=num_epochs)
    assert len(monitor.epoch) == num_epochs
    assert all([monitor.epoch[i] == i + 1 for i in range(num_epochs)])
    assert len(list(iter(monitor))) == 1  # "dummy"
    assert np.allclose(
        monitor.dummy.mean,
        np.mean(
            [
                np.arange(1, num_iters + 1)
                for num_iters in monitor._iterations_per_epoch
            ]
        ),
    )
    assert np.allclose(
        monitor.dummy.std,
        np.std(
            [
                np.arange(1, num_iters + 1)
                for num_iters in monitor._iterations_per_epoch
            ]
        ),
    )
    assert np.allclose(
        monitor.dummy.percentiles,
        np.percentile(
            [
                np.arange(1, num_iters + 1)
                for num_iters in monitor._iterations_per_epoch
            ],
            np.array([0.0, 100.0]),
        ),
    )
