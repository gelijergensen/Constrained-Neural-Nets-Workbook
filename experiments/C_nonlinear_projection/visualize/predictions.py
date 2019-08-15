"""Plotting of the model predictions in the input and output space"""

import matplotlib.pyplot as plt

from .config import DEFAULT_DIRECTORY
from .readability_utils import _correct_and_clean_labels

__all__ = ["plot_model_predictions"]


def plot_model_predictions(
    inputs,
    outputs,
    prediction_sets,
    labels,
    savefile,
    title="Model Predictions",
    xlabel="Input",
    ylabel="Output",
    directory=DEFAULT_DIRECTORY,
):
    """

    :param inputs: numpy array of inputs to model (not including any 
        parameterization)
    :param outputs: numpy array of outputs of model
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
    clean_labels = _correct_and_clean_labels(labels)

    # Plot the datapoints
    fig = plt.figure()
    plt.plot(inputs, outputs, "-.", color="black")
    for i, (predictions, label) in enumerate(
        zip(prediction_sets, clean_labels)
    ):
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
