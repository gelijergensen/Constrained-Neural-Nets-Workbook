"""Tools for visualizing the results of the experiment"""

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_DIRECTORY = "/global/u1/g/gelijerg/Projects/pyinsulate/results"


def _clean_labels(labels):
    """Replaces all underscores with spaces and capitalizes first letter"""

    corrected_labels = [
        (label if label != "step_optimizer" else "backward_pass")
        for label in labels
    ]
    return [label.replace("_", " ").capitalize() for label in corrected_labels]


def plot_loss(
    monitors,
    labels,
    savefile,
    constrained=False,
    title="Losses",
    ylabel="Average loss",
    directory=DEFAULT_DIRECTORY,
):
    """Plots several loss curves

    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param constrained: whether to plot the constrained or unconstrained loss.
        Defaults to unconstrained
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    epochs = monitors[0].epoch

    all_mean_losses = np.zeros((len(epochs), len(labels)))
    for i, monitor in enumerate(monitors):
        losses = (
            monitor.mean_loss if not constrained else monitor.constrained_loss
        )
        this_batch_size = monitor.batch_size

        all_mean_losses[:, i] = np.array(
            [
                np.average(loss, weights=batch_size)
                for loss, batch_size in zip(losses, this_batch_size)
            ]
        )

    fig = plt.figure(figsize=(4, 3))
    plt.plot(epochs, all_mean_losses)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend(labels)

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving loss plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig


def plot_constraints(
    monitors,
    labels,
    savefile,
    title="Constraint magnitude",
    ylabel="Average constraint magnitude",
    directory=DEFAULT_DIRECTORY,
):
    """Plots the magnitude of the constraints

    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    epochs = monitors[0].epoch

    all_mean_constraints = np.zeros((len(epochs), len(labels)))
    for i, monitor in enumerate(monitors):
        constraints = monitor.constraints
        batch_sizes = monitor.batch_size

        all_mean_constraints[:, i] = np.array(
            [
                np.average(
                    [torch.norm(con).item() for con in constraint],
                    weights=batch_size,
                )
                for constraint, batch_size in zip(constraints, batch_sizes)
            ]
        )

    fig = plt.figure(figsize=(4, 3))
    plt.plot(epochs, all_mean_constraints)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend(labels)

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving constraint satisfaction plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig


def plot_time(
    monitor,
    savefile,
    title="Average computation time per batch",
    xlabel="Milliseconds",
    directory=DEFAULT_DIRECTORY,
):
    """Plots the computation time required for each step as a horizontal bar 
    plot

    :param monitor: the training monitor
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param xlabel: label for the x-axis
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    time_keys = np.array(monitor.time_keys)
    times = np.array([np.mean(monitor.get(key)) for key in time_keys])

    idxs = np.argsort(times)
    times = times[idxs]
    time_keys = time_keys[idxs]
    labels = _clean_labels(time_keys)

    fig = plt.figure(figsize=(4, 3))
    plt.barh(labels, times, tick_label=["" for _ in labels])
    plt.xlabel(xlabel)
    plt.title(title)

    x = (plt.xlim()[1] - plt.xlim()[0]) * 0.02
    for y, label in enumerate(labels):
        plt.text(x, y, label, ha="left", va="center")
    plt.yticks([], direction="in")

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving compute time plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig


def plot_model_predictions(
    data,
    prediction_sets,
    labels,
    savefile,
    title="Model Predictions",
    xlabel="Input",
    ylabel="Output",
    directory=DEFAULT_DIRECTORY,
):
    """

    :param data: a tuple (inputs, outputs, is_training) where is_training
        is a boolean mask for which data samples were training data
    :param prediction_sets: a list of a list of model predictions (can plot
        multiple models)
    :param labels: a list of the labels for each model
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    inputs, outputs, training_mask = data

    # Plot the datapoints
    fig = plt.figure(figsize=(4, 3))
    plt.plot(
        inputs[training_mask], outputs[training_mask], "x", color="#555555"
    )
    plt.plot(
        inputs[np.logical_not(training_mask)],
        outputs[np.logical_not(training_mask)],
        "x",
        color="#BBBBBB",
    )
    for i, (predictions, label) in enumerate(zip(prediction_sets, labels)):
        plt.plot(inputs, prediction_sets[i], "--", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving loss plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig

