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
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots several loss curves

    :param monitors: a list of pairs of monitors [(training, evaluation)].
        Elements of the pair can be set to None to be skipped
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param constrained: whether to plot the constrained or unconstrained loss.
        Defaults to unconstrained
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    def get_data(monitor):
        if monitor is not None:
            losses = (
                monitor.mean_loss
                if not constrained
                else monitor.constrained_loss
            )
            this_batch_size = monitor.batch_size

            return np.array(
                [
                    np.average(loss, weights=batch_size)
                    for loss, batch_size in zip(losses, this_batch_size)
                ]
            )
        else:
            return None

    epochs = None

    fig = plt.figure()

    for i, (monitor_pair, label) in enumerate(zip(monitors, labels)):
        training_monitor, evaluation_monitor = monitor_pair

        color = None
        if training_monitor is not None:
            training_data = get_data(training_monitor)
            line2d = plt.plot(
                training_monitor.epoch,
                training_data,
                "-",
                label=f"{label} (Training)",
            )
            color = line2d[0].get_color()
        if evaluation_monitor is not None:
            evaluation_data = get_data(evaluation_monitor)
            plt.plot(
                evaluation_monitor.epoch,
                evaluation_data,
                "--",
                label=f"{label} (Evaluation)",
                color=color,
            )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend()

    # possibly make log plot
    if log:
        plt.yscale("log")

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(
            f"Saving {'constrained ' if constrained else ''}loss plot to {filepath}"
        )
        plt.savefig(filepath, dpi=300)
    return fig


def plot_constraints(
    monitors,
    labels,
    savefile,
    title="Constraint magnitude",
    ylabel="Average constraint magnitude",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the magnitude of the constraints

    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    def get_data(monitor):
        if monitor is not None:
            constraints = monitor.constraints
            batch_sizes = monitor.batch_size

            return np.array(
                [
                    np.average(
                        [torch.norm(con).item() for con in constraint],
                        weights=batch_size,
                    )
                    for constraint, batch_size in zip(constraints, batch_sizes)
                ]
            )
        else:
            return None

    epochs = None

    fig = plt.figure()

    for i, (monitor_pair, label) in enumerate(zip(monitors, labels)):
        training_monitor, evaluation_monitor = monitor_pair

        color = None
        if training_monitor is not None:
            training_data = get_data(training_monitor)
            line2d = plt.plot(
                training_monitor.epoch,
                training_data,
                "-",
                label=f"{label} (Training)",
            )
            color = line2d[0].get_color()
        if evaluation_monitor is not None:
            evaluation_data = get_data(evaluation_monitor)
            plt.plot(
                evaluation_monitor.epoch,
                evaluation_data,
                "--",
                label=f"{label} (Evaluation)",
                color=color,
            )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend()

    # possibly make log plot
    if log:
        plt.yscale("log")

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving constraint satisfaction plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig


def plot_time(
    monitors,
    labels,
    savefile,
    title="Average computation time per batch",
    xlabel="Milliseconds",
    time_keys=None,
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the computation time required for each step as a horizontal bar 
    plot

    :param monitors: a list of monitors
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param xlabel: label for the x-axis
    :param time_keys: time keys to plot. Defaults to all available
    :param log: whether to plot on a log scale
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    # Get a complete list of time_keys:
    if time_keys is None:
        time_keys = set()
        for monitor in monitors:
            print(_clean_labels(monitor.timing[0][0].keys()))
            time_keys.update(
                [
                    key
                    for key in monitor.timing[0][0].keys()
                    if "error" not in key and "recomputed" not in key
                ]
            )
        time_keys = list(time_keys)

    all_average_times = list()
    for monitor in monitors:
        batch_sizes = np.array(monitor.get("batch_size"))
        # only the batches with a full batch size (assume first batch is full)
        valid_observations = batch_sizes == batch_sizes.ravel()[0]

        average_times = list()
        for key in time_keys:
            if key not in monitor.timing[0][0]:
                average_times.append(0.0)
                continue
            times = np.array(
                [
                    [batch_time.get(key, -999.0) for batch_time in epoch_times]
                    for epoch_times in monitor.timing
                ]
            )
            # average ignoring invalid observations and flag values of -999.0
            average_times.append(
                np.nan_to_num(
                    np.mean(
                        times[np.logical_and(valid_observations, times > -998)]
                    )
                )
            )
        all_average_times.append(average_times)
    all_average_times = np.array(all_average_times)

    max_heights = np.amax(all_average_times, axis=0)
    idxs = np.argsort(max_heights)
    all_average_times = all_average_times[:, idxs]
    time_keys = np.array(time_keys)[idxs]
    clean_time_keys = _clean_labels(time_keys)

    # do normalization if requested

    # Using the recipe for a grouped bar plot
    fig = plt.figure()

    # set width of bars
    bar_width = 1.0 / (1.0 + len(all_average_times))
    for i, (average_times, label) in enumerate(zip(all_average_times, labels)):

        positions = np.arange(len(clean_time_keys)) + i * bar_width

        plt.barh(positions, average_times, height=bar_width, label=label)
    # # Add divider bars
    # positions = (
    #     np.arange(len(clean_time_keys) + 1)
    #     # + len(all_average_times) * bar_width
    #     - 0.25
    # )
    # for pos in positions:
    #     plt.axhline(pos, color="grey")

    # Add ticks on the middle of the group bars
    ys = (
        # np.linspace(0.0, 1.0, len(clean_time_keys) + 1)
        np.arange(len(clean_time_keys))
        + 0.5 * len(all_average_times) * bar_width
        - 0.5 * bar_width
    )
    plt.yticks(ys, clean_time_keys, ha="left", va="center")

    ax = plt.gca()
    ax.tick_params(axis="y", direction="in", pad=-5)
    labels = ax.yaxis.get_ticklabels()
    for label in labels:
        label.set_bbox(
            {
                "facecolor": "white",
                "edgecolor": "white",
                "alpha": 0.45,
                "pad": 1.00,
            }
        )
    plt.legend(loc="lower right")

    # possibly make log plot
    if log:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.title(title)

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
    fig = plt.figure()
    plt.plot(
        inputs[training_mask], outputs[training_mask], "o", color="#555555"
    )
    plt.plot(
        inputs[np.logical_not(training_mask)],
        outputs[np.logical_not(training_mask)],
        "o",
        color="#BBBBBB",
    )
    for i, (predictions, label) in enumerate(zip(prediction_sets, labels)):
        plt.plot(inputs, predictions, "-", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving prediction plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig

