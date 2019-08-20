"""An experiment to evaluate the efficacy of constrained neural network training
methods and draw comparisons"""

import functools
from ignite.engine import Events
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pyinsulate.ignite import GradientConstraint, GradientLoss

from .checkpointer import ModelAndMonitorCheckpointer
from .constraints import helmholtz_equation, pythagorean_equation
from .dataloader import get_multiwave_dataloaders
from .event_loop import create_engine, Sub_Batch_Events
from .model import Dense, ParameterizedDense
from .monitor import TrainingMonitor, ProjectionMonitor
from .monitor_predictions import PredictionLogger


__all__ = ["run_experiment", "default_configuration"]


def default_configuration():
    """Default configuration for the experiment. Recognized kwargs:
    seed: seed to use for all random number generation
    training_parameterizations: dictionary of parameters for training data. See
        dataloader.get_multiwave_dataloaders() for more details
    testing_parameterizations: dictionary of parameters for testing data. See
        dataloader.get_multiwave_dataloaders() for more details
    batch_size: batch size. Defaults to 100
    projection_batch_size: batch size for projection. Defaults to same as 
        batch_size
    model_size: a list of integers for the lengths of the layers of the
        model. Defaults to [20].
    model_act: activation function for the model. Defaults to nn.Tanh()
    model_final_act: activation function for last layer. Defaults to None
    learning_rate: learning rate. Defaults to 0.01
    projection_learning_rate: learning rate for projection. Defaults to same
        as learning_rate
    device: device to run on ("cpu"/"cuda"). Defaults to "cpu"
    regularization_weight: multiplier to use for soft-constraints during
        training. Defaults to 0, for unconstrained
    constraint: function to use for constraining during projection
    error_fn: error function to use for converting the constraint function to an
        error function for soft constraining. Defaults to MSE
    tolerance: desired maximum value of constraint error
    max_iterations: maximum number of iterations in the projection step
    """
    return {
        "seed": None,
        "training_parameterizations": {
            "amplitudes": [1.0],
            "frequencies": [1.0],
            "phases": [0.0],
            "num_points": 20,
            "sampling": "uniform",
        },
        "testing_parameterizations": {
            "amplitudes": [1.0],
            "frequencies": [1.0],
            "phases": [0.0],
            "num_points": 20,
            "sampling": "uniform",
        },
        "batch_size": 10,
        "projection_batch_size": None,
        "architecture": Dense,
        "model_size": [20],
        "model_act": nn.Tanh(),
        "model_final_act": None,
        "learning_rate": 0.01,
        "projection_learning_rate": None,
        "device": "cpu",
        "regularization_weight": 0,
        "constraint": helmholtz_equation,
        "error_fn": None,
        "tolerance": 1e-5,
        "max_iterations": 1e4,
    }


def get_data(configuration):
    """Grabs the training and testing dataloaders for this configuration"""
    proj_batch_size = (
        configuration["batch_size"]
        if configuration["projection_batch_size"] is None
        else configuration["projection_batch_size"]
    )
    return get_multiwave_dataloaders(
        configuration["training_parameterizations"],
        configuration["testing_parameterizations"],
        seed=configuration["seed"],
        batch_size=configuration["batch_size"],
        proj_batch_size=proj_batch_size,
    )


def build_model_and_optimizer(configuration):
    """Creates the model, optimizer"""
    architecture = configuration["architecture"]
    model = architecture(
        1,  # dimension of input
        3,  # dimension of parameters
        1,  # dimension of output
        sizes=configuration["model_size"],
        activation=configuration["model_act"],
        final_activation=configuration["model_final_act"],
    ).to(device=torch.device(configuration["device"]))
    opt = optim.Adam(model.parameters(), lr=configuration["learning_rate"])
    proj_lr = (
        configuration["learning_rate"]
        if configuration["projection_learning_rate"] is None
        else configuration["projection_learning_rate"]
    )
    proj_opt = optim.Adam(model.parameters(), lr=proj_lr)
    return model, opt, proj_opt


def get_loss_and_constraint(configuration):
    """Retrieves the loss and constraint"""
    # We need the entire batch of losses, not it's sum
    loss = nn.MSELoss(reduction="none")
    constraint = configuration["constraint"]
    return loss, constraint


def run_experiment(
    max_epochs,
    log=None,
    evaluate=True,
    projection=True,
    save_directory=".",
    save_file=None,
    save_interval=1,
    **configuration,
):
    """Runs the Proof of Constraint experiment with the given configuration

    :param max_epochs: number of epochs to run the experiment
    :param log: function to use for logging. None supresses logging
    :param evaluate: whether to run the evaluator once over the 
        training data at the end of an epoch
    :param projection: whether to run the projection engine once over the 
        testing data at the end of an epoch
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
    train_dl, test_dl = get_data(kwargs)

    # Build the model, optimizer, loss, and constraint
    model, opt, proj_opt = build_model_and_optimizer(kwargs)
    loss, constraint = get_loss_and_constraint(kwargs)

    # Setup Monitors and Checkpoints
    training_monitor = TrainingMonitor("training")
    evaluation_monitor = TrainingMonitor("evaluation") if evaluate else None
    projection_monitor = ProjectionMonitor() if projection else None
    prediction_logger = PredictionLogger(model)
    if should_checkpoint:
        checkpointer = ModelAndMonitorCheckpointer(
            save_directory,
            save_file,
            kwargs,
            [training_monitor, evaluation_monitor, projection_monitor],
            prediction_logger,
            save_interval=save_interval,
        )
    else:
        checkpointer = None

    # This is the trainer because we provide the optimizer
    trainer = create_engine(
        model,
        loss,
        constraint,
        opt,
        projection=False,
        monitor=training_monitor,
        regularization_weight=kwargs["regularization_weight"],
        error_fn=kwargs["error_fn"],
        device=kwargs["device"],
        tolerance=kwargs["tolerance"],
        max_iterations=kwargs["max_iterations"],
    )

    # These are not trainers simply because we don't provide the optimizer
    if evaluate:
        evaluator = create_engine(
            model,
            loss,
            constraint,
            optimizer=None,
            projection=False,
            monitor=evaluation_monitor,
            regularization_weight=kwargs["regularization_weight"],
            error_fn=kwargs["error_fn"],
            device=kwargs["device"],
            tolerance=kwargs["tolerance"],
            max_iterations=kwargs["max_iterations"],
        )
    else:
        evaluator = None
    if projection:
        projector = create_engine(
            model,
            loss,
            constraint,
            proj_opt,
            projection=True,
            monitor=projection_monitor,
            regularization_weight=kwargs["regularization_weight"],
            error_fn=kwargs["error_fn"],
            device=kwargs["device"],
            tolerance=kwargs["tolerance"],
            max_iterations=kwargs["max_iterations"],
        )
    else:
        projector = None

    prediction_logger.attach(trainer, projector)

    # Ensure evaluation happens once per epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluation(trainer):
        if training_monitor is not None and should_log:
            summary = training_monitor.summarize()
            log(
                f"Epoch[{trainer.state.epoch:05d}] Training Summary - {summary}"
            )

        if evaluate:
            if should_log:
                log(
                    f"Epoch[{trainer.state.epoch:05d}] - Evaluating on training data..."
                )
            evaluator.run(train_dl)
            if evaluation_monitor is not None and should_log:
                summary = evaluation_monitor.summarize()
                log(
                    f"Epoch[{trainer.state.epoch:05d}] Evaluation Summary - {summary}"
                )

        # Handle projection
        if projection:
            if should_log:
                log(f"Epoch[{trainer.state.epoch:05d}] - Projecting...")
            projector.run(test_dl, max_epochs=kwargs["max_iterations"])
            if projection_monitor is not None and should_log:
                summary = projection_monitor.summarize()
                log(
                    f"Epoch[{trainer.state.epoch:05d}] Generalization Summary - {summary}"
                )

        if should_checkpoint:
            checkpointer(trainer)

    # Handle projection summary
    if projection:

        @projector.on(Events.EPOCH_COMPLETED)
        def projection_summary(projector):
            if projection_monitor is not None and should_log:
                summary = projection_monitor.summarize(during_projection=True)
                log(
                    f"Epoch[{trainer.state.epoch:05d}-{projector.state.epoch:05d}] Projection Summary - {summary}"
                )

        @projector.on(Events.EPOCH_COMPLETED)
        def projection_stop(projector):
            if projection_monitor is not None:
                if projection_monitor.should_stop_projection(
                    kwargs["tolerance"]
                ):
                    projector.terminate()

        @projector.on(Events.COMPLETED)
        def projection_unterminate(projector):
            # Unblock the projector so it can resume later
            projector.should_terminate = False

    if should_log:

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_batch_summary(trainer):
            log(
                "Epoch[{:05d}] - Total loss: {:.5f}, Data Loss: {:.5f}, Constraint Error: {:.5f}".format(
                    trainer.state.epoch,
                    trainer.state.total_loss.cpu().item(),
                    trainer.state.mean_loss.cpu().item(),
                    trainer.state.constraints_error.cpu().item(),
                )
            )

    trainer.run(train_dl, max_epochs=max_epochs)

    # Save final model and monitors
    if should_checkpoint:
        checkpointer.retrieve_and_save(trainer)

    return (
        kwargs,
        (trainer, evaluator, projector),
        (training_monitor, evaluation_monitor, projection_monitor),
    )

