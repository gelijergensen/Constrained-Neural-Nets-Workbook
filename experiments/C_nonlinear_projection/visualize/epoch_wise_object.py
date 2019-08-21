"""Tools for plotting single values for each epoch"""

import matplotlib.pyplot as plt

from .config import DEFAULT_DIRECTORY
from .readability_utils import _clean_label, _correct_and_clean_labels
from .retrieval_utils import retrieve_object


__all__ = ["plot_epoch_wise"]


def plot_epoch_wise(
    xvalues,
    yvalues,
    labels,
    savefile,
    colors=None,
    line_styles=None,
    title="Untitled",
    xlabel="Epoch",
    ylabel="Unspecified",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the data epoch-wise for each monitor

    :param xvalues: a list of lists. Probably [x.epoch for x in monitors]
    :param yvalues: a list of lists
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param colors: a list of colors or None. If a sorted list of integers is 
        given, then it is interpreted as identities for colors
    :param line_styles: a list of line_styles or None
    :param title: title of the figure. Defaults to "Untitled"
    :param ylabel: label for the y-axis. Defaults to "Unspecified"
    :param xlabel: label for the x-axis. Defaults to "Epoch"
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    stashed_colors = list()
    if colors is None:
        colors = [None for _ in labels]
    if line_styles is None:
        line_styles = ["-" for _ in labels]
    clean_labels = _correct_and_clean_labels(labels)

    fig = plt.figure()
    for xvalue, yvalue, label, color, line_style in zip(
        xvalues, yvalues, clean_labels, colors, line_styles
    ):
        if color is None:
            plt.plot(xvalue, yvalue, line_style, label=label)
        elif isinstance(color, int):
            if len(stashed_colors) > color:
                color = stashed_colors[color]
                plt.plot(xvalue, yvalue, line_style, label=label, color=color)
            else:
                line2d = plt.plot(xvalue, yvalue, line_style, label=label)
                color = line2d[0].get_color()
                stashed_colors.append(color)
        else:
            plt.plot(xvalue, yvalue, line_style, label=label, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
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
        print(f"Saving {title} plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig
