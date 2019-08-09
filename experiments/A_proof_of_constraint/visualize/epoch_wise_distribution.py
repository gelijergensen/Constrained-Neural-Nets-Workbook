"""Tools for plotting complete distributions for each object"""

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_DIRECTORY
from .readability_utils import _clean_label, _correct_and_clean_labels
from .retrieval_utils import retrieve_object


__all__ = ["plot_constraints_distribution", "plot_parameters_distribution"]


def _plot_object_distribution(
    monitors,
    labels,
    savefile,
    object_string,
    retrieval_kwargs=dict(),
    plot_iterations=False,
    title=None,
    ylabel=None,
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots a distribution with several curves for each monitor for the given 
    object string

    :param monitors: a list of pairs of monitors [(training, evaluation)].
        Elements of the pair can be set to None to be skipped
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param object_string: string for the object to retreive. See 
        retrieval_utils.retrieve_object for more details
    :param retrieval_kwargs: dictionary of any necessary kwargs for the 
        retrieval process. See retrieval_utils.retrieve_object for more details
    :param plot_iterations: whether to plot by iteration, rather than by epoch.
        Will assume the data is of shape (epoch, batch, ...) and flatten to
        (total_iterations, ...)
    :param title: title of the figure. Defaults to a cleaned version of the
        object string
    :param ylabel: label for the y-axis. Defaults to a cleaned version of 
        f"Average {object_string}"
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    clean_labels = _correct_and_clean_labels(labels)

    if title is None:
        title = _clean_label(object_string)
    if ylabel is None:
        ylabel = f"Average {_clean_label(object_string)}"

    fig = plt.figure()
    for i, (monitor_set, label) in enumerate(zip(monitors, clean_labels)):

        if len(monitor_set) == 1:
            suffixes = [""]
        elif len(monitor_set) == 2:
            suffixes = [" (Training)", " (Evaluation)"]
        else:
            suffixes = ["SET SUFFIXES!" for monitor in monitor_set]

        for monitor, suffix in zip(monitor_set, suffixes):
            if monitor is None:
                continue
            data = retrieve_object(monitor, object_string, **retrieval_kwargs)
            if plot_iterations:
                # flatten the batch axis into the epoch axis
                data = data.reshape(-1, data.shape[2:])
                xvalues = np.array(monitor.iteration).ravel()
            else:
                xvalues = np.array(monitor.epoch)
            # Retrieve all 100 percentiles
            percentiles = np.percentile(
                data,
                np.linspace(0, 100, num=101),  # Must be odd
                axis=1,  # 0th axis is epoch
            )
            midpoint = int((len(percentiles) - 1) / 2)
            line2d = plt.plot(
                xvalues,
                percentiles[midpoint],
                "-",
                label=f"{label}{suffix}",
                zorder=10,
            )
            color = line2d[0].get_color()

            for i in range(midpoint):
                plt.fill_between(
                    xvalues,
                    percentiles[i],
                    percentiles[-i - 1],
                    color=color,
                    alpha=(
                        5.0 / (len(percentiles) - 1)
                    ),  # This requires >5 percentiles!
                    zorder=0,
                )
            plt.plot(xvalues, percentiles[0], ":", color=color, zorder=10)
            plt.plot(xvalues, percentiles[-1], ":", color=color, zorder=10)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Iteration" if plot_iterations else "Epoch")
    l = plt.legend()
    l.set_zorder(20)

    # possibly make log plot
    if log:
        if log == "symlog":
            plt.yscale("symlog")
        else:
            plt.yscale("log")

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving {object_string} distribution plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig


def plot_constraints_distribution(
    monitors,
    labels,
    savefile,
    absolute_value=False,
    title="Constraint",
    ylabel="Constraint value",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the distribution of the constraints

    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param absolute_value: whether to plot the absolute value of the constraints
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    object_string = "constraints"
    return _plot_object_distribution(
        monitors,
        labels,
        savefile,
        object_string,
        retrieval_kwargs={
            "distribution": True,
            "absolute_value": absolute_value,
        },
        title=title,
        ylabel=ylabel,
        log=log,
        directory=directory,
    )


def plot_parameters_distribution(
    monitors,
    labels,
    savefile,
    gradients=False,
    title="Parameters",
    ylabel="Parameter values",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the distribution of the constraints

    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param gradients: whether to plot the gradients of the parameters
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    object_string = "model_parameters"
    return _plot_object_distribution(
        monitors,
        labels,
        savefile,
        object_string,
        retrieval_kwargs={"gradients": gradients},
        title=title,
        ylabel=ylabel,
        log=log,
        directory=directory,
    )
