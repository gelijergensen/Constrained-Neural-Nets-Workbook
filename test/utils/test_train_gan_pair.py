import torch
import torch.nn as nn
from torch import optim
from ..load_test_data import get_mnist_gan_dataloaders

from pyinsulate.utils.train_gan_pair import gan_batched_losses, train_gan_pair


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


def test_gan_batched_losses():

    train_dl, valid_dl = get_mnist_gan_dataloaders(batch_size=128)
    gener = Mnist_Generator()
    discr = Mnist_Discriminator()
    loss_fn = nn.BCELoss()
    gener_opt = optim.SGD(gener.parameters(), lr=0.1)
    discr_opt = optim.SGD(discr.parameters(), lr=0.1)

    for xb, yb in train_dl:
        gener.train()
        discr.train()

        gen_orig_params = next(x.data.clone().detach()
                               for x in gener.parameters())
        dis_orig_params = next(x.data.clone().detach()
                               for x in discr.parameters())

        gan_batched_losses(gener, discr, loss_fn, None,
                           xb, yb, gener_opt, discr_opt)

        gen_new_params = next(x.data.clone().detach()
                              for x in gener.parameters())
        dis_new_params = next(x.data.clone().detach()
                              for x in discr.parameters())
        assert(not torch.equal(gen_orig_params, gen_new_params))
        assert(not torch.equal(dis_orig_params, dis_new_params))

        gan_batched_losses(gener, discr, loss_fn, None, xb, yb)

        gen_still_new_params = next(x.data.clone().detach()
                                    for x in gener.parameters())
        dis_still_new_params = next(x.data.clone().detach()
                                    for x in discr.parameters())
        assert(torch.equal(gen_new_params, gen_still_new_params))
        assert(torch.equal(dis_new_params, dis_still_new_params))
        break


def test_train_model():

    train_dl, valid_dl = get_mnist_gan_dataloaders(batch_size=128)
    gener = Mnist_Generator()
    discr = Mnist_Discriminator()
    gener_opt = optim.SGD(gener.parameters(), lr=0.1)
    discr_opt = optim.SGD(discr.parameters(), lr=0.1)

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

    train_gan_pair(gener, discr, train_dl, valid_dl,
                   gener_opt, discr_opt, until, log=log)

    assert(log.epoch == 2)
