"""An experiment to evaluate the efficacy of constrained neural network training
methods and draw comparisons"""

import functools
from ignite.engine import Events
import torch
import torch.nn as nn
import torch.optim as optim

from .checkpointer import ModelAndMonitorCheckpointer
from .constraints import helmholtz_equation, pythagorean_equation
from .dataloader import get_multiwave_dataloaders
from .event_loop import create_engine, Sub_Batch_Events
from .model import Dense, ParameterizedDense
from .monitor import ProofOfConstraintMonitor


__all__ = ["run_experiment", "default_configuration"]


def default_configuration():
    """Default configuration for the experiment. Recognized kwargs:
    seed: seed to use for all random number generation
    training_parameterizations: dictionary of parameters for training data. See
        dataloader.get_multiwave_dataloaders() for more details
    testing_parameterizations: dictionary of parameters for testing data. See
        dataloader.get_multiwave_dataloaders() for more details
    batch_size: batch size. Defaults to 100
    model_size: a list of integers for the lengths of the layers of the
        model. Defaults to [20].
    model_act: activation function for the model. Defaults to nn.Tanh()
    model_final_act: activation function for last layer. Defaults to None
    learning_rate: learning rate. Defaults to 0.01
    device: device to run on ("cpu"/"cuda"). Defaults to "cpu"
    method: method to use for constraining. See the event loop for more details
    constraint: function to use for constraining
    reduction: reduction to use for constraining. See event loop for details
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
        "architecture": Dense,
        "model_size": [20],
        "model_act": nn.Tanh(),
        "model_final_act": None,
        "learning_rate": 0.01,
        "device": "cpu",
        "method": "constrained",
        "constraint": helmholtz_equation,
        "reduction": None,
    }


def get_data(configuration):
    """Grabs the training and testing dataloaders for this configuration"""
    return get_multiwave_dataloaders(
        configuration["training_parameterizations"],
        configuration["testing_parameterizations"],
        seed=configuration["seed"],
        batch_size=configuration["batch_size"],
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
    return model, opt


def get_loss_and_constraint(configuration):
    """Retrieves the loss and constraint"""
    # We need the entire batch of losses, not it's sum
    loss = nn.MSELoss(reduction="none")
    constraint = configuration["constraint"]
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
    train_dl, test_dl = get_data(kwargs)

    # Setup Monitors and Checkpoints
    training_monitor = ProofOfConstraintMonitor()
    evaluation_train_monitor = (
        ProofOfConstraintMonitor(is_evaluation=True)
        if evaluate_training
        else None
    )
    evaluation_test_monitor = (
        ProofOfConstraintMonitor(is_evaluation=True)
        if evaluate_testing
        else None
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
        device=kwargs["device"],
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
            device=kwargs["device"],
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
            device=kwargs["device"],
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
                    trainer.state.constrained_loss.cpu().item(),
                    trainer.state.mean_loss.cpu().item(),
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

