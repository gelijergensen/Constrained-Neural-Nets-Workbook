"""An experiment to evaluate the efficacy of constrained neural network training
methods and draw comparisons"""

import functools
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
import torch
import torch.nn as nn
import torch.optim as optim

from pyinsulate.ignite import GradientConstraint, GradientLoss
from pyinsulate.lagrange.exact import constrain_loss
from pyinsulate.losses.pdes import helmholtz_equation

from .dataloader import get_singlewave_dataloaders
from .model import Dense


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options."""
    return tuple(convert_tensor(x, device=device, non_blocking=non_blocking) for x in batch)


def create_trainer(model, optimizer, loss_fn, constraint_fn, **constraint_kwargs):

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        xb, yb = prepare_batch(batch)
        out = model(xb)
        last = getattr(engine.state, "last", None)
        if last is not None and len(out) == len(last) and torch.allclose(out, last):
            print("WARNING! Just outputting same thing!")
        engine.state.last = out
        if torch.allclose(out, out.new_zeros(out.size())):
            print("WARNING! Training is failing")

        loss = loss_fn(out, yb)
        constraints = constraint_fn(out, xb, **constraint_kwargs)

        constrained_loss = constrain_loss(
            loss, constraints, list(model.parameters()))
        constrained_loss.backward()
        optimizer.step()

        engine.state.constraints = constraints
        engine.state.loss = loss
        return constrained_loss.item()

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


def default_configuration():
    """Default configuration for the experiment. Recognized kwargs:
    frequency: frequency of the wave equation. Defaults to 1.0
    phase: phase of the wave_equation. Defaults to None (random)
    amplitude: amplitude of the wave equation. Defaults to 1.0
    num_points: number of points to evaluate
    num_training: number of training datapoints
    training_sampling: one of
        "start" - provide the first num_training points as training
        "uniform" - provide num_training points distributed evenly across the
            domain
        "random" - randomly sample points for training
    batch_size: batch size. Defaults to 32
    model_size: a list of integers for the lengths of the layers of the
        model. Defaults to [20].
    model_act: activation function for the model. Defaults to nn.ReLU()
    model_final_act: activation function for last layer. Defaults to None
    learning_rate: learning rate. Defaults to 0.01
    """
    return {
        'frequency': 1.0,
        'phase': None,
        'amplitude': 1.0,
        'num_points': 100000,
        'num_training': 100,
        'training_sampling': "start",
        'batch_size': 32,
        'model_size': [20],
        'model_act': nn.ReLU(),
        'model_final_act': None,
        'learning_rate': 0.01,
    }


def abs_value_decorator(fn):
    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return torch.abs(fn(*args, **kwargs))
    return decorated


def mean_absolute_value_decorator(fn):
    """Take mean of abs along batch dimension"""
    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return torch.mean(torch.abs(fn(*args, **kwargs)), dim=0)
    return decorated


def run_experiment(max_epochs, logging=False, **configuration):
    """Runs the Proof of Constraint experiment with the given configuration

    :param max_epochs: number of epochs to run the experiment
    :param logging: whether to log to the terminal
    :param configuration: kwargs for various settings. See get_configuration
        for more details
    """
    kwargs = default_configuration()
    kwargs.update(configuration)

    train_dl, test_dl = get_singlewave_dataloaders(
        frequency=kwargs['frequency'], phase=kwargs['phase'], amplitude=kwargs['amplitude'],
        num_points=kwargs['num_points'], num_training=kwargs['num_training'], sampling=kwargs['training_sampling'],
        batch_size=kwargs['batch_size']
    )

    model = Dense(1, 1, sizes=kwargs['model_size'],
                  activation=kwargs['model_act'],
                  final_activation=kwargs['model_final_act'])
    opt = optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
    loss = nn.MSELoss()
    constraint = abs_value_decorator(helmholtz_equation)

    trainer = create_trainer(
        model, opt, loss, constraint, k=kwargs['frequency'])
    evaluator = create_evaluator(model, metrics={
        'mse':  GradientLoss(loss, output_transform=lambda args: (args[2], args[1])),
        'constraint':
            GradientConstraint(
                constraint,
                output_transform=lambda args: (
                    args[2], args[0], {'k': kwargs['frequency']})
        )
    })

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        evaluator.run(test_dl)
        if logging:
            metrics = evaluator.state.metrics
            summary = ""
            for key in metrics:
                summary += f"{key}: {metrics[key]}\t"
            print(summary)

    if logging:
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            print("Epoch[{}] - Constrained loss: {:.5f}, Loss: {:.5f}, Constraint: {}".format(
                trainer.state.epoch, trainer.state.output, trainer.state.loss, trainer.state.constraints))

    trainer.run(train_dl, max_epochs=max_epochs)
    return (evaluator.state.metrics['mse'], evaluator.state.metrics['constraint'])
