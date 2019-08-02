"""An experiment to evaluate the efficacy of constrained neural network training
methods and draw comparisons"""

import functools
from ignite.engine import Events
import torch
import torch.nn as nn
import torch.optim as optim

from pyinsulate.ignite import GradientConstraint, GradientLoss
from pyinsulate.pdes import helmholtz_equation

from .checkpointer import ModelAndMonitorCheckpointer
from .dataloader import get_singlewave_dataloaders
from .event_loop import create_engine, Sub_Batch_Events
from .model import Dense
from .monitor import ProofOfConstraintMonitor


__all__ = ["run_experiment", "default_configuration"]


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
    method: method to use for constraining. See the event loop for more details
    """
    return {
        "frequency": 1.0,
        "phase": None,
        "amplitude": 1.0,
        "num_points": 100000,
        "num_training": 100,
        "training_sampling": "start",
        "batch_size": 32,
        "model_size": [20],
        "model_act": nn.ReLU(),
        "model_final_act": None,
        "learning_rate": 0.01,
        "method": "constrained",
        "reduction": None,
        "ground_approximation": None,
    }


def get_data(configuration, return_equation=False):
    """Grabs the training and testing dataloaders for this configuration"""
    return get_singlewave_dataloaders(
        frequency=configuration["frequency"],
        phase=configuration["phase"],
        amplitude=configuration["amplitude"],
        num_points=configuration["num_points"],
        num_training=configuration["num_training"],
        sampling=configuration["training_sampling"],
        batch_size=configuration["batch_size"],
        seed=configuration.get("seed", None),
        return_equation=return_equation,
    )


def build_model_and_optimizer(configuration):
    """Creates the model, optimizer"""
    model = Dense(
        1,
        1,
        sizes=configuration["model_size"],
        activation=configuration["model_act"],
        final_activation=configuration["model_final_act"],
    )
    opt = optim.Adam(model.parameters(), lr=configuration["learning_rate"])
    return model, opt


def get_loss_and_constraint(configuration):
    """Retrieves the loss and constraint"""
    # We need the entire batch of losses, not it's sum
    loss = nn.MSELoss(reduction="none")
    constraint = helmholtz_equation
    return loss, constraint


def run_experiment(
    max_epochs,
    log=None,
    evaluate_training=True,
    evaluate_testing=True,
    save_directory=".",
    save_file=None,
    save_interval=1,
    **configuration,
):
    """Runs the Proof of Constraint experiment with the given configuration

    :param max_epochs: number of epochs to run the experiment
    :param log: function to use for logging. None supresses logging
    :param evaluate_training: whether to run the evaluator once over the 
        training data at the end of an epoch. Will be overridden if 
        evaluation_train_monitor is provided
    :param evaluate_testing: whether to run the evaluator once over the 
        testing data at the end of an epoch. Will be overridden if
        evaluation_test_monitor is provided
    :param save_directory: optional directory to save checkpoints into. Defaults
        to the directory that the main script was called from
    :param save_file: base filename for checkpointing. If not provided, then no
        checkpointing will be performed
    :param save_interval: frequency of saving out model checkpoints. Defaults to
        every epoch
    :param configuration: kwargs for various settings. See default_configuration
        for more details
    :returns: the configuration dictionary, a tuple of all engines (first will
        be the training engine), and a corresponding tuple of all monitors
    """

    # Determine the parameters of the analysis
    should_log = log is not None
    should_checkpoint = save_file is not None
    kwargs = default_configuration()
    kwargs.update(configuration)
    if should_log:
        log(kwargs)

    # Get the data
    train_dl, test_dl, parameterization = get_data(kwargs)
    kwargs.update(
        parameterization
    )  # ensures any random variables are specified

    # Setup Monitors and Checkpoints
    training_monitor = ProofOfConstraintMonitor()
    evaluation_train_monitor = (
        ProofOfConstraintMonitor() if evaluate_training else None
    )
    evaluation_test_monitor = (
        ProofOfConstraintMonitor() if evaluate_testing else None
    )
    if should_checkpoint:
        checkpointer = ModelAndMonitorCheckpointer(
            save_directory,
            save_file,
            kwargs,
            [
                training_monitor,
                evaluation_train_monitor,
                evaluation_test_monitor,
            ],
            save_interval=save_interval,
        )
    else:
        checkpointer = None

    # Build the model, optimizer, loss, and constraint
    model, opt = build_model_and_optimizer(kwargs)
    loss, constraint = get_loss_and_constraint(kwargs)

    # This is the trainer because we provide the optimizer
    trainer = create_engine(
        model,
        loss,
        constraint,
        opt,
        monitor=training_monitor,
        method=kwargs["method"],
        reduction=kwargs["reduction"],
        ground_approximation=kwargs["ground_approximation"],
        k=kwargs["frequency"],
    )

    # These are not trainers simply because we don't provide the optimizer
    if evaluate_training:
        train_evaluator = create_engine(
            model,
            loss,
            constraint,
            monitor=evaluation_train_monitor,
            method=kwargs["method"],
            reduction=kwargs["reduction"],
            ground_approximation=kwargs["ground_approximation"],
            k=kwargs["frequency"],
        )
    else:
        train_evaluator = None
    if evaluate_testing:
        test_evaluator = create_engine(
            model,
            loss,
            constraint,
            monitor=evaluation_test_monitor,
            method=kwargs["method"],
            reduction=kwargs["reduction"],
            ground_approximation=kwargs["ground_approximation"],
            k=kwargs["frequency"],
        )
    else:
        test_evaluator = None

    # Ensure evaluation happens once per epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluation(trainer):
        if training_monitor is not None and should_log:
            summary = training_monitor.summarize()
            log(
                f"Epoch[{trainer.state.epoch}] Training (Training) Summary - {summary}"
            )

        if evaluate_training:
            if should_log:
                log(
                    f"Epoch[{trainer.state.epoch}] - Evaluating on training data..."
                )
            train_evaluator.run(train_dl)
            if evaluation_train_monitor is not None and should_log:
                summary = evaluation_train_monitor.summarize()
                log(
                    f"Epoch[{trainer.state.epoch}] Evaluation (Training) Summary - {summary}"
                )

        if evaluate_testing:
            if should_log:
                log(
                    f"Epoch[{trainer.state.epoch}] - Evaluating on testing data..."
                )
            test_evaluator.run(test_dl)
            if evaluation_test_monitor is not None and should_log:
                summary = evaluation_test_monitor.summarize()
                log(
                    f"Epoch[{trainer.state.epoch}] Evaluation (Testing) Summary  - {summary}"
                )

        if should_checkpoint:
            checkpointer(trainer)

    if should_log:

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_batch_summary(trainer):
            log(
                "Epoch[{}] - Constrained loss: {:.5f}, Loss: {:.5f}".format(
                    trainer.state.epoch,
                    trainer.state.constrained_loss,
                    trainer.state.mean_loss,
                )
            )

    trainer.run(train_dl, max_epochs=max_epochs)

    # Save final model and monitors
    if should_checkpoint:
        checkpointer.retrieve_and_save(trainer)

    return (
        kwargs,
        (trainer, train_evaluator, test_evaluator),
        (training_monitor, evaluation_train_monitor, evaluation_test_monitor),
    )

