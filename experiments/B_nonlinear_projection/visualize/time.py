"""Plotting of the time it takes for various computation steps"""

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_DIRECTORY
from .readability_utils import _correct_and_clean_labels


__all__ = ["plot_time"]


def plot_time(
    monitors,
    labels,
    savefile,
    title="Average computation time per epoch",
    ylabel="Seconds",
    log=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots the computation time required for each step as a horizontal bar 
    plot

    :param monitors: a list of monitor sets: [(training, evaluation, inference)]
    :param labels: a list of strings for the label of each monitor
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param ylabel: label for the y-axis
    :param log: whether to plot a log-plot. Can also be set to "symlog"
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """
    clean_labels = _correct_and_clean_labels(labels)
    all_times = np.array(
        [
            [
                np.mean(
                    [
                        np.sum(epoch["total"])
                        for epoch in training_monitor.timing
                    ]
                ),
                np.mean(
                    [
                        np.sum([iteration["total"] for iteration in epoch])
                        for epoch in projection_monitor.timing
                    ]
                ),
            ]
            for (
                training_monitor,
                evaluation_monitor,
                projection_monitor,
            ) in monitors
        ]
    )

    # Using the recipe for a grouped bar plot
    fig = plt.figure()
    # set width of bars
    bar_width = 1.0 / (1.0 + all_times.shape[1])
    colors = list()
    for i, times in enumerate(all_times):
        positions = bar_width * np.arange(len(times)) + i

        for j, (position, time, label) in enumerate(
            zip(positions, times, ["Training", "Projection"])
        ):
            if i == 0:
                line2d = plt.bar(position, time, width=bar_width, label=label)
                colors.append(line2d[0].get_facecolor())
            else:
                plt.bar(position, time, width=bar_width, color=colors[j])

    # Add ticks on the middle of the group bars
    xs = (
        np.arange(len(all_times))
        + 0.5 * all_times.shape[1] * bar_width
        - 0.5 * bar_width
    )
    plt.xticks(xs, clean_labels)

    plt.legend()

    # possibly make log plot
    if log:
        if log == "symlog":
            plt.yscale("symlog")
        else:
            plt.yscale("log")

    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving timing plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig

