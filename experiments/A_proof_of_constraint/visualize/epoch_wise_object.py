"""Tools for visualizing the results of the experiment"""

import matplotlib.pyplot as plt

from .config import DEFAULT_DIRECTORY
from .readability_utils import _clean_label, _correct_and_clean_labels
from .retrieval_utils import retrieve_object


__all__ = ["plot_loss", "plot_constraints", "plot_constraints_diagnostics"]


def _plot_object(
    monitors,
    labels,
    savefile,
    object_string,
    retrieval_kwargs=dict(),
    title=None,
    ylabel=None,
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots several curves for each monitor for the given object string

    :param monitors: a list of pairs of monitors [(training, evaluation)].
        Elements of the pair can be set to None to be skipped
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param object_string: string for the object to retreive. See 
        retrieval_utils.retrieve_object for more details
    :param retrieval_kwargs: dictionary of any necessary kwargs for the 
        retrieval process. See retrieval_utils.retrieve_object for more details
    :param title: title of the figure. Defaults to a cleaned version of the
        object string
    :param ylabel: label for the y-axis. Defaults to a cleaned version of 
        f"Average {object_string}"
    :param log: whether to plot a log-plot
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    clean_labels = _correct_and_clean_labels(labels)

    if title is None:
        title = _clean_label(object_string)
    if ylabel is None:
        ylabel = f"Average {_clean_label(object_string)}"

    fig = plt.figure()
    for i, (monitor_pair, label) in enumerate(zip(monitors, clean_labels)):
        training_monitor, evaluation_monitor = monitor_pair

        color = None
        if training_monitor is not None:
            training_data = retrieve_object(
                training_monitor, object_string, **retrieval_kwargs
            )
            line2d = plt.plot(
                training_monitor.epoch,
                training_data,
                "-",
                label=f"{label} (Training)",
            )
            color = line2d[0].get_color()
        if evaluation_monitor is not None:
            evaluation_data = retrieve_object(
                evaluation_monitor, object_string, **retrieval_kwargs
            )
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
        print(f"Saving {object_string} plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig


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
    if constrained:
        object_string = "constrained_loss"
    else:
        object_string = "mean_loss"
    return _plot_object(
        monitors,
        labels,
        savefile,
        object_string,
        title=title,
        ylabel=ylabel,
        log=log,
        directory=directory,
    )


def plot_constraints(
    monitors,
    labels,
    savefile,
    absolute_value=False,
    reduced=False,
    title="Constraint",
    ylabel="Average constraint value",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the magnitude of the constraints

    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param absolute_value: whether to plot the absolute value of the constraints
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    if reduced:
        object_string = "reduced_constraints"
    else:
        object_string = "constraints"
    return _plot_object(
        monitors,
        labels,
        savefile,
        object_string,
        retrieval_kwargs={"absolute_value": absolute_value},
        title=title,
        ylabel=ylabel,
        log=log,
        directory=directory,
    )


def plot_constraints_diagnostics(
    monitors,
    labels,
    savefile,
    diagnostics_index,
    title="Constraint magnitude",
    ylabel="Average constraint magnitude",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the magnitude of the constraints
    
    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param diagnostics_index: index of the diagnostics tensor to retreive. See 
        the PDE for more details
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    object_string = "constraints_diagnostics"
    return _plot_object(
        monitors,
        labels,
        savefile,
        object_string,
        retrieval_kwargs={"index": diagnostics_index},
        title=title,
        ylabel=ylabel,
        log=log,
        directory=directory,
    )
