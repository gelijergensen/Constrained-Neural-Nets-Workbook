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

__all__ = ["run_experiment"]


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options."""
    return tuple(convert_tensor(x, device=device, non_blocking=non_blocking) for x in batch)


def create_trainer(model, optimizer, loss_fn, constraint_fn, metrics=None, **constraint_kwargs):

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
        engine.state.constrained_loss = constrained_loss
        return xb, yb, out

    engine = Engine(_update)

    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(engine, name)

    return engine


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
        return fn(*args, **kwargs)
        # return torch.mean(torch.abs(fn(*args, **kwargs)), dim=0)
    return decorated


def run_experiment(max_epochs, log=None, training_monitor=None, evaluation_train_monitor=None, evaluation_test_monitor=None, evaluate_training=True, evaluate_testing=True, **configuration):
    """Runs the Proof of Constraint experiment with the given configuration

    :param max_epochs: number of epochs to run the experiment
    :param log: function to use for logging. None supresses logging
    :param training_monitor: function to use for monitoring during training. 
        Should be of the form trainer -> void. Will be called every epoch end
    :param evaluation_train_monitor: same as training_monitor, but for 
        evaluation on the training data
    :param evaluation_test_monitor: same as training_monitor, but for 
        evaluation on the testing data
    :param evaluate_training: whether to run the evaluator once over the 
        training data at the end of an epoch. Will be overridden if 
        evaluation_train_monitor is provided
    :param evaluate_testing: whether to run the evaluator once over the 
        testing data at the end of an epoch. Will be overridden if
        evaluation_test_monitor is provided
    :param configuration: kwargs for various settings. See default_configuration
        for more details
    """
    evaluate_training &= evaluation_train_monitor is not None
    evaluate_testing &= evaluation_test_monitor is not None

    kwargs = default_configuration()
    kwargs.update(configuration)

    should_log = log is not None

    if should_log:
        log(kwargs)

    train_dl, test_dl, equation = get_singlewave_dataloaders(
        frequency=kwargs['frequency'], phase=kwargs['phase'], amplitude=kwargs['amplitude'],
        num_points=kwargs['num_points'], num_training=kwargs['num_training'], sampling=kwargs['training_sampling'],
        batch_size=kwargs['batch_size'], return_equation=True
    )

    model = Dense(1, 1, sizes=kwargs['model_size'],
                  activation=kwargs['model_act'],
                  final_activation=kwargs['model_final_act'])
    opt = optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
    loss = nn.MSELoss()
    constraint = abs_value_decorator(helmholtz_equation)

    trainer = create_trainer(
        model, opt, loss, constraint, k=kwargs['frequency']
    )

    if evaluate_training:
        train_evaluator = create_evaluator(model, metrics={
            'loss':  GradientLoss(loss, output_transform=lambda args: (args[2], args[1])),
            'constraint':
            GradientConstraint(
                constraint,
                output_transform=lambda args: (
                    args[2], args[0], {'k': kwargs['frequency']})
            )
        })
    if evaluate_testing:
        test_evaluator = create_evaluator(model, metrics={
            'loss':  GradientLoss(loss, output_transform=lambda args: (args[2], args[1])),
            'constraint':
            GradientConstraint(
                constraint,
                output_transform=lambda args: (
                    args[2], args[0], {'k': kwargs['frequency']})
            )
        })

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluation(trainer):
        if training_monitor is not None:
            training_monitor(trainer)

        if evaluate_training:
            train_evaluator.run(test_dl)
            if should_log:
                metrics = train_evaluator.state.metrics
                summary = f"Epoch[{trainer.state.epoch}] Training Summary - "
                for key in metrics:
                    summary += f"{key}: {metrics[key]}\t"
                log(summary)
            if evaluation_train_monitor is not None:
                evaluation_train_monitor(train_evaluator)

        if evaluate_testing:
            test_evaluator.run(test_dl)
            if should_log:
                metrics = test_evaluator.state.metrics
                summary = f"Epoch[{trainer.state.epoch}] Testing Summary - "
                for key in metrics:
                    summary += f"{key}: {metrics[key]}\t"
                log(summary)
            if evaluation_test_monitor is not None:
                evaluation_test_monitor(test_evaluator)

    if should_log:
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_batch_summary(trainer):
            log("Epoch[{}] - Constrained loss: {:.5f}, Loss: {:.5f}".format(
                trainer.state.epoch, trainer.state.constrained_loss, trainer.state.loss))

    trainer.run(train_dl, max_epochs=max_epochs)
    return train_dl, test_dl, equation
