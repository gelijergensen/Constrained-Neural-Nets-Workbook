"""Tools which are important for the retrieval of data to visualize, but aren't
part of the visualization process itself"""

import numpy as np
import torch

__all__ = ["retrieve_object"]


def retrieve_object(monitor, object_string, **kwargs):
    """Retrives the requested data from the given monitor

    :param monitor: either a training or evaluation monitor
    :param object_string: string to identify the object to retrieve. Should be
        one of "mean_loss", "total_loss", "constraints_error", 
        "constraints", "model_parameters"
    :param kwargs: all other options passed to downstream methods

    :returns: a numpy array of the data. If average_batches, then has shape
        (num_epochs,). Otherwise has shape (num_epochs, num_batchs_per_epoch)
    """
    if object_string in [
        "mean_loss",
        "total_loss",
        "constraints_error",
        "projection_epochs",
    ]:
        data = retrieve_plain(monitor, object_string, **kwargs)
    elif object_string == "constraints":
        data = retrieve_constraint(monitor, **kwargs)
    # elif object_string == "constraints_diagnostics":
    #     data = retrieve_constraints_diagnostics(monitor, **kwargs)
    elif object_string == "model_parameters":
        data = retrieve_parameters(monitor, **kwargs)
    else:
        raise ValueError(f"{object_string} not recognized for data retrieval")

    return data


def retrieve_plain(monitor, object_string):
    """Retrieves the request object as-is (doesn't apply any modification). This
    is valid only for objects which are single values (items from a tensor)

    :param monitor: either a training or evaluation monitor
    :param object_string: string to identify the object to retrieve
    :returns: the data for the given monitor directly
    """
    return getattr(monitor, object_string)


def retrieve_constraint(monitor, absolute_value=False):
    """Retrieves the percentiles of the constraint (or the absolute value of
    the ith-constriant). WARNING: assumes only one constraint
    
    :param monitor: either a training or evaluation monitor
    :param absolute_value: whether to take the absolute value of the constraint
    :returns: the percentiles of the ith-constraint
    """
    if absolute_value:
        object_string = "constraints_abs_percentiles"
    else:
        object_string = "constraints_percentiles"
    return retrieve_plain(monitor, object_string)


def retrieve_parameters(monitor, gradients=False, differences=False):
    """Retrieves the parameters of the model or their gradients

    :param monitor: either a training or evaluation monitor
    :param gradients: whether to retrieve the gradients instead of the values
    :param differences: whether to retrieve the differences in parameters from
        original to final
    :returns: the values of the parameters (or gradients) for each batch and 
        epoch
    """
    if differences:
        if not hasattr(monitor, "model_parameters_difference_percentiles"):
            raise AttributeError(
                "Monitor is not projection monitor, so cannot retrieve model parameter differences"
            )
        data = monitor.model_parameters_difference_percentiles
    else:
        if gradients:
            data = monitor.model_parameters_grad_percentiles
            if data[0][0] is None:
                raise AttributeError("Parameters do not have gradients")
        else:
            data = monitor.model_parameters_percentiles

    return data

