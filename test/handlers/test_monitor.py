from ignite.engine import Events, create_supervised_trainer
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


class EpochMonitor(Monitor):

    def __init__(self):
        super().__init__()

        self.add('epoch')
        self.add('avg_epoch', average=True)

    def __call__(self, engine):
        epoch = float(engine.state.epoch)
        self.set('epoch', epoch)
        self.set('avg_epoch', epoch)


def test_monitor():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    trainer = create_supervised_trainer(model, opt, loss_fn)

    epoch_logger = EpochMonitor()
    epoch_logger.attach(trainer)

    num_epochs = 1
    trainer.run(train_dl, max_epochs=num_epochs)
    assert(len(epoch_logger.epoch) == num_epochs)
    assert(all([all(epoch == i+1 for epoch in epoch_logger.epoch[i])
                for i in range(num_epochs)]))
    assert(all(epoch == i+1 for i, epoch in enumerate(epoch_logger.avg_epoch)))
