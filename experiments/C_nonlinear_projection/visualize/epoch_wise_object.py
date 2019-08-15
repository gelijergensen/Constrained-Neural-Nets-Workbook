"""Tools for plotting single values for each epoch"""

import matplotlib.pyplot as plt

from .config import DEFAULT_DIRECTORY
from .readability_utils import _clean_label, _correct_and_clean_labels
from .retrieval_utils import retrieve_object


__all__ = [
    "plot_loss",
    "plot_constraints_error",
]  # , "plot_constraints", "plot_constraints_diagnostics"]


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
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    possible_line_styles = ["-", "--", "-.", ":", "."]

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

        color = None
        for monitor, suffix, line_style in zip(
            monitor_set, suffixes, possible_line_styles
        ):
            if monitor is None:
                continue
            data = retrieve_object(monitor, object_string, **retrieval_kwargs)

            if color is None:
                line2d = plt.plot(
                    monitor.epoch, data, "-", label=f"{label}{suffix}"
                )
                color = line2d[0].get_color()
            else:
                plt.plot(
                    monitor.epoch,
                    data,
                    line_style,
                    label=f"{label}{suffix}",
                    color=color,
                )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend()

    # possibly make log plot
    if log:
        if log == "symlog":
            plt.yscale("symlog")
        else:
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
    total=False,
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
    :param total: whether to plot the constrained or unconstrained loss.
        Defaults to unconstrained
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    if total:
        object_string = "total_loss"
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


def plot_constraints_error(
    monitors,
    labels,
    savefile,
    title="Constraint error",
    ylabel="Average constraint value",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the constraints error, as if it were a loss

    :param monitors: a list of monitors, e.g. [training, evaluation]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    object_string = "constraints_error"
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
