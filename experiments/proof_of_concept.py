import functools
from ignite.engine import Engine, Events
# from ignite.metrics import Loss
from ignite.utils import convert_tensor
import numpy as np
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pyinsulate.ignite import GradientLoss
from pyinsulate.losses import steady_state_turbulence


import time


def load_data(filepath, num_training, num_testing=None, batch_size=32, just_datasets=False):
    if num_testing is None:
        num_testing = 128*128*128 - num_training
    all_indices = np.random.permutation(128*128*128)
    train_idxs = all_indices[:num_training]
    test_idxs = all_indices[num_training:num_training+num_testing]

    train_set = ProofOfConceptDataset(filepath, train_idxs)
    test_set = ProofOfConceptDataset(filepath, test_idxs)

    if just_datasets:
        return train_set, test_set
    else:
        return create_dataloaders(train_set, test_set, batch_size=batch_size)


def create_dataloaders(train_set, test_set, batch_size=32):
    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_dl = DataLoader(test_set, batch_size=batch_size)

    return train_dl, test_dl


class ProofOfConceptDataset(Dataset):

    # static data cache
    cached_data = dict()

    def __init__(self, filepath, indices, data=None):
        if data is None:
            self.data = self.load_data(filepath)
        else:
            self.data = data
        self.indices = indices

    def clone(self):
        """Produces a clone which doesn't share the underlying data. Useful for
        multiple processes"""
        data = tuple(x.clone() for x in self.data)
        return ProofOfConceptDataset(None, self.indices, data=data)

    def load_data(self, filepath):
        """Caches the data so that multiple datasets can share the data"""
        if filepath in self.cached_data:
            return self.cached_data[filepath]

        data = np.load(filepath).astype(np.float32)
        velocity = torch.from_numpy(data).view(-1, 3)
        position = torch.stack(torch.meshgrid(
            [
                torch.linspace(-0.5, 0.5, 128),
                torch.linspace(-0.5, 0.5, 128),
                torch.linspace(-0.5, 0.5, 128)
            ]
        ), dim=-1).view(-1, 3)

        self.cached_data[filepath] = (position, velocity)
        return (position, velocity)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Force the model to try and memorize only a few points
        # idx = idx % 50
        return self.data[0][idx].requires_grad_(), self.data[1][idx]


class PINN(nn.Module):
    """A model which is standard dense neural network, realized either as a
    dense neural network or as a series of 1x1 convolutional filters"""

    @staticmethod
    def get_layer(in_size, out_size, conv=False):
        if conv:
            return nn.Conv1d(in_size, out_size, 1)
        else:
            return nn.Linear(in_size, out_size)

    def __init__(self, in_size, out_size, sizes=None, convolutional=False, activation=nn.LeakyReLU(0.01)):
        super().__init__()
        if sizes is None:
            sizes = [20, 20, 20, 20, 20]
        self.conv = convolutional
        self.act = activation
        self.layer0 = self.get_layer(in_size, sizes[0], conv=self.conv)
        for i in range(1, len(sizes)):
            setattr(self, f'layer{i}',
                    self.get_layer(sizes[i-1], sizes[i], conv=self.conv))
        setattr(self, f'layer{len(sizes)}',
                self.get_layer(sizes[-1], out_size, conv=self.conv))

        self.layers = [getattr(self, f'layer{i}') for i in range(len(sizes)+1)]

    def forward(self, xb):

        if self.conv:
            # Add final dummy final "space" dimension
            xb = xb.unsqueeze(-1)
            # xb = xb.view(1, *(xb.size()[1:]), xb.size()[0])
        for layer in self.layers:
            xb = self.act(layer(xb))
        xb = self.act(xb)
        if self.conv:
            # Convert convolution dimension back to batch dimension
            # also remove dummy dimension 0
            # xb = xb.view(xb.size()[-1], *(xb.size()[1:-1]))
            # remove dummy dimension -1
            xb = xb.view(xb.size()[:-1])
        return xb


class ResPINN(nn.Module):
    """A model which is standard dense neural network, realized either as a
    dense neural network or as a series of 1x1 convolutional filters. This
    version contains several residual blocks"""

    @staticmethod
    def get_layer(in_size, out_size, conv=False):
        if conv:
            return nn.Conv1d(in_size, out_size, 1)
        else:
            return nn.Linear(in_size, out_size)

    def __init__(self, in_size, out_size, sizes=None, convolutional=False, activation=nn.LeakyReLU(0.01)):
        super().__init__()
        if sizes is None:
            sizes = [[20, 20, 20], [20, 20, 20]]
        self.conv = convolutional
        self.act = activation
        first_size = in_size
        self.blocks = list()
        for b, block in enumerate(sizes):
            setattr(self, f'block{b}_layer0',
                    self.get_layer(first_size, block[0], conv=self.conv))
            first_size = out_size
            for i in range(1, len(block)):
                setattr(self, f'block{b}_layer{i}',
                        self.get_layer(block[i-1], block[i], conv=self.conv))
            setattr(self, f'block{b}_layer{len(block)}',
                    self.get_layer(block[-1], out_size, conv=self.conv))
            self.blocks.append(
                [getattr(self, f'block{b}_layer{i}') for i in range(len(block)+1)])

    def forward(self, xb):

        if self.conv:
            # Add final dummy final "space" dimension
            xb = xb.unsqueeze(-1)
            # xb = xb.view(1, *(xb.size()[1:]), xb.size()[0])
        for layer in self.blocks[0]:
            xb = self.act(layer(xb))
        out = xb
        for block in self.blocks[1:]:
            for l, layer in enumerate(block):
                if l == 0:
                    xb = self.act(layer(out))
                else:
                    xb = self.act(layer(xb))
            out = out + xb
        xb = self.act(out)
        if self.conv:
            # Convert convolution dimension back to batch dimension
            # also remove dummy dimension 0
            # xb = xb.view(xb.size()[-1], *(xb.size()[1:-1]))
            # remove dummy dimension -1
            xb = xb.view(xb.size()[:-1])
        return xb


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options."""
    return tuple(convert_tensor(x, device=device, non_blocking=non_blocking) for x in batch)


def create_trainer(model, optimizer, loss_fn, pde_loss_fn=None):

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        xb, yb = prepare_batch(batch)
        out = model(xb)
        last = getattr(engine.state, "last", None)
        if last is not None and torch.allclose(out, last):
            print("WARNING! Just outputting same thing!")
        engine.state.last = out
        if torch.allclose(out, out.new_zeros(out.size())):
            print("WARNING! Training is failing")
        pde_loss = 0 if pde_loss_fn is None else pde_loss_fn(out, xb)
        loss = loss_fn(out, yb)
        total_loss = loss + pde_loss
        total_loss.backward()
        optimizer.step()

        engine.state.pde_loss = pde_loss
        engine.state.loss = loss
        return total_loss.item()

    return Engine(_update)


def create_evaluator(model, metrics):

    def _inference(engine, batch):
        model.eval()
        with torch.enable_grad():  # we need the gradient for the metrics
            xb, yb = prepare_batch(batch)
            out = model(xb)
        return xb, yb, out

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def run_analysis(train_dl, test_dl, sizes, activation, logging=False):
    try:
        iter(sizes[0])
        model_fn = ResPINN
    except Exception:
        # not iterable
        model_fn = PINN
    model = model_fn(3, 3, convolutional=False,
                     sizes=sizes, activation=activation)
    opt = optim.Adam(model.parameters(), lr=0.01)
    pde_loss = steady_state_turbulence
    loss = nn.MSELoss()

    trainer = create_trainer(model, opt, loss, pde_loss_fn=pde_loss)
    evaluator = create_evaluator(
        model, metrics={
            'pde': GradientLoss(pde_loss, output_transform=lambda args: (args[2], args[0])),
            'mse': GradientLoss(loss, output_transform=lambda args: (args[2], args[1]))
        }
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        evaluator.run(test_dl)
        if logging:
            metrics = evaluator.state.metrics
            print(metrics['pde'], metrics['mse'])

    if logging:
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            print("Epoch[{}] - Total loss: {:.5f} = PDE Loss: {:.5f} + MSE: {:.5f}".format(
                trainer.state.epoch, trainer.state.output, trainer.state.pde_loss, trainer.state.loss))

    trainer.run(train_dl, max_epochs=20)
    return (evaluator.state.metrics['pde'], evaluator.state.metrics['mse'])


if __name__ == "__main__":
    path = os.path.expandvars('$SCRATCH/data/divfree-test/raw_0100.npy')
    train_dl, test_dl = load_data(
        path, 32*32*32, num_testing=128, batch_size=128)

    # model = PINN(3, 3, convolutional=False, sizes=[
    #              20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20
    #              ],
    #              activation=nn.LeakyReLU())
    # model = ResPINN(3, 3, convolutional=False, sizes=[
    #     [20, 20, 20], [20, 20, 20], [20, 20, 20], [20, 20, 20]
    # ], activation=nn.LeakyReLU(0.01))
    # print(model)
    # opt = optim.Adam(model.parameters(), lr=0.01)
    # pde_loss = steady_state_turbulence
    # loss = nn.MSELoss()

    final_loss = run_analysis(train_dl, test_dl, [20], nn.LeakyReLU())
    print(final_loss)
    print('done!')

    # # Test PDE Loss
    # times = list()
    # print("Testing loss (actual)")
    # for xb, yb in train_dl:
    #     out = model(xb)
    #     t0 = time.time()
    #     pde_loss(out, xb)
    #     t1 = time.time()
    #     times.append(t1 - t0)
    # print(
    #     f"It took {sum(times)} ({sum(times)/float(len(times))}) seconds for one epoch (iteration)")

    # trainer = create_trainer(model, opt, loss, pde_loss_fn=pde_loss)
    # evaluator = create_evaluator(
    #     model, metrics={
    #         'pde': GradientLoss(pde_loss, output_transform=lambda args: (args[2], args[0])),
    #         'mse': GradientLoss(loss, output_transform=lambda args: (args[2], args[1]))
    #     }
    # )

    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(trainer):
    #     print("Epoch[{}] - Total loss: {:.5f} = PDE Loss: {:.5f} + MSE: {:.5f}".format(
    #         trainer.state.epoch, trainer.state.output, trainer.state.pde_loss, trainer.state.loss))

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(trainer):
    #     print("Evaluating on training...")
    #     evaluator.run(train_dl)
    #     metrics = evaluator.state.metrics
    #     print("Validation Results: Epoch[{}] - PDE Loss: {:.5f} + MSE: {:.5f}"
    #           .format(trainer.state.epoch, metrics['pde'], metrics['mse']))

    # too computationally expensive for now
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(trainer):
    #     print("Evaluating on validation...")
    #     evaluator.run(test_dl)
    #     metrics = evaluator.state.metrics
    #     print("Validation Results: Epoch[{}] - PDE Loss: {:.5f} + MSE: {:.5f}"
    #           .format(trainer.state.epoch, metrics['pde'], metrics['mse']))

    # trainer.run(train_dl, max_epochs=200)

    # print('done!')
