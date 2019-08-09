"""Plotting of the time it takes for various computation steps"""

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_DIRECTORY
from .readability_utils import _correct_and_clean_labels


__all__ = ["plot_time", "plot_time_experiment"]


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
    :param log: whether to plot a log-plot. Can also be set to "symlog"
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
        if log == "symlog":
            plt.xscale("symlog")
        else:
            plt.xscale("log")

    plt.xlabel(xlabel)
    plt.title(title)

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving compute time plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig


def plot_time_experiment(
    monitor_groups,
    labels,
    xvalues,
    savefile,
    title="Time per iteration",
    ylabel="Time per iteration",
    xlabel="Unspecified x value",
    confidence_interval=95.0,
    normalize=False,
    directory=DEFAULT_DIRECTORY,
):
    """Plots a curve for computation time required for different configurations

    :param monitor_groups: a list of lists of monitors, where the outer grouping
        is by monitor label
    :param labels: a list of strings for the label of each monitor group
    :param xvalues: xvalues to plot against (e.g. the batch sizes)
    :param savefile: name of the file to save. If none, then will not save
    :param title: title of the figure
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param confidence: how wide of a confidence interval to plot, where 100.0
        will plot the [min, max] interval
    :param normalize: whether to normalize each curve to its initial value
    :param directory: directory to save the file in. Defaults to the results dir
    :returns: the figure
    """

    clean_labels = _correct_and_clean_labels(labels)
    # retrieve the data
    ci_min = (100.0 - confidence_interval) / 2.0
    ci_max = 100.0 - ci_min

    grouped_means = list()
    grouped_cis = list()
    normalizations = list()
    for monitors in monitor_groups:
        means = list()
        cis = list()
        # We assume the monitors are already sorted for us
        for monitor in monitors:
            # only first epoch
            batch_sizes = np.array(monitor.batch_size[0])
            # only batches with full batch size
            valid_observations = batch_sizes == batch_sizes[0]

            times = np.array(
                [batch_time.get("total") for batch_time in monitor.timing[0]]
            )
            means.append(np.mean(times[valid_observations]))
            cis.append(
                np.percentile(times[valid_observations], [ci_min, ci_max])
            )
        means = np.array(means)
        cis = np.array(cis)
        if normalize:
            factor = means[0]
            means /= factor
            cis /= factor

        grouped_means.append(means)
        grouped_cis.append(cis)
    grouped_means = np.array(grouped_means)
    grouped_cis = np.array(grouped_cis)

    fig = plt.figure()
    for mean, cis, label in zip(grouped_means, grouped_cis, clean_labels):
        line2d = plt.plot(xvalues, mean, label=label, zorder=10)
        color = line2d[0].get_color()
        plt.fill_between(
            xvalues, cis[:, 0], cis[:, 1], color=color, alpha=0.5, zorder=0
        )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    l = plt.legend()
    l.set_zorder(20)

    plt.tight_layout()

    if savefile is not None:
        filepath = f"{directory}/{savefile}.png"
        print(f"Saving timing experiment plot to {filepath}")
        plt.savefig(filepath, dpi=300)
    return fig
