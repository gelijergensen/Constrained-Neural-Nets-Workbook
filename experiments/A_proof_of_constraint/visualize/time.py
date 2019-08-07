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
    clean_time_keys = _correct_and_clean_labels(time_keys)

    # Using the recipe for a grouped bar plot
    fig = plt.figure()
    # set width of bars
    bar_width = 1.0 / (1.0 + len(all_average_times))
    for i, (average_times, label) in enumerate(zip(all_average_times, labels)):

        positions = np.arange(len(clean_time_keys)) + i * bar_width

        plt.barh(positions, average_times, height=bar_width, label=label)
    # Add ticks on the middle of the group bars
    ys = (
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

