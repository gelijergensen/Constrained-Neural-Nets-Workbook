"""Tools for plotting complete distributions for each object"""

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_DIRECTORY
from .readability_utils import _clean_label, _correct_and_clean_labels
from .retrieval_utils import retrieve_object


__all__ = ["plot_epoch_wise_distribution"]


def plot_epoch_wise_distribution(
    monitors,
    labels,
    data,
    savefile,
    colors=None,
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
    :param data: a list of 2d numpy arrays, where the first dimension is by
        epochs and the second is different percentiles
    :param savefile: name of the file to save. If none, then will not save
    :param colors: a list of colors or None. If a sorted list of integers is 
        given, then it is interpreted as identities for colors
    :param title: title of the figure. Defaults to "Untitled"
    :param ylabel: label for the y-axis. Defaults to "Unspecified"
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    stashed_colors = list()
    if colors is None:
        colors = [None for _ in monitors]
    clean_labels = _correct_and_clean_labels(labels)
    if title is None:
        title = "Untitled"
    if ylabel is None:
        ylabel = "Unspecified"

    fig = plt.figure()
    for monitor, label, datum, color in zip(
        monitors, clean_labels, data, colors
    ):
        percentiles = np.array(datum)
        xvalues = np.array(monitor.epoch)
        midpoint = int((percentiles.shape[1] - 1) / 2)
        if color is None:
            line2d = plt.plot(
                xvalues, percentiles[:, midpoint], "-", label=label, zorder=10
            )
            color = line2d[0].get_color()
        elif isinstance(color, int):
            if len(stashed_colors) > color:
                color = stashed_colors[color]
                plt.plot(
                    xvalues,
                    percentiles[:, midpoint],
                    "-",
                    label=label,
                    color=color,
                    zorder=10,
                )
            else:
                line2d = plt.plot(
                    xvalues,
                    percentiles[:, midpoint],
                    "-",
                    label=label,
                    zorder=10,
                )
                color = line2d[0].get_color()
                stashed_colors.append(color)
        else:
            plt.plot(
                xvalues,
                percentiles[:, midpoint],
                "-",
                label=label,
                color=color,
                zorder=10,
            )

        for i in range(midpoint):
            plt.fill_between(
                xvalues,
                percentiles[:, i],
                percentiles[:, -i - 1],
                color=color,
                alpha=(
                    5.0 / (percentiles.shape[1] - 1)
                ),  # This requires >5 percentiles!
                zorder=0,
            )
        plt.plot(xvalues, percentiles[:, 0], ":", color=color, zorder=10)
        plt.plot(xvalues, percentiles[:, -1], ":", color=color, zorder=10)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
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
        print(f"Saving {title} distribution plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig
