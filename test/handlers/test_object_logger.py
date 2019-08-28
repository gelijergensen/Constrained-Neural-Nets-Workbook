from ignite.engine import Events, create_supervised_trainer
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from ..load_test_data import get_mnist_dataloaders

from src.handlers import ObjectLogger


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def test_object_logger():

    train_dl, valid_dl = get_mnist_dataloaders(batch_size=256)
    model = Mnist_Logistic()
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=0.1)

    trainer = create_supervised_trainer(model, opt, loss_fn)

    def get_epoch(engine):
        return engine.state.epoch

    epoch_logger = ObjectLogger(trainer, get_epoch)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, epoch_logger)

    trainer.run(train_dl, max_epochs=1)

    assert len(epoch_logger.values) == 1
