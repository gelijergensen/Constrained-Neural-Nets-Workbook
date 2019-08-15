"""Tools which are important for the retrieval of data to visualize, but aren't
part of the visualization process itself"""

import numpy as np
import torch


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
    if object_string in ["mean_loss", "constraints_error"]:
        data = retrieve_plain(monitor, object_string)
    elif object_string in ["total_loss"]:
        data = retrieve_total_loss(monitor)
    elif object_string == "loss":
        data = retrieve_loss(monitor, **kwargs)
    elif object_string == "constraints":
        data = retrieve_constraint(monitor, **kwargs)
    # elif object_string == "constraints_diagnostics":
    #     data = retrieve_constraints_diagnostics(monitor, **kwargs)
    elif object_string == "model_parameters":
        data = retrieve_parameters(monitor, **kwargs)
    else:
        raise ValueError(f"{object_string} not recognized for data retrieval")

    return data


def retrieve_plain(monitor, object_string, original=True):
    """Retrieves the request object as-is (doesn't apply any modification). This
    is valid only for objects which are single values (items from a tensor)

    :param monitor: either a training or evaluation monitor
    :param object_string: string to identify the object to retrieve
    :param original: if the monitor is an evaluation monitor, retrieve
        the original object instead of the final object
    :returns: the data for the given monitor directly
    """
    if monitor.is_evaluation:
        if original:
            true_object_string = f"original_{object_string}"
        else:
            true_object_string = f"final_{object_string}"
    else:
        true_object_string = object_string

    return getattr(monitor, true_object_string)


def retrieve_total_loss(monitor):
    """Retrieves the total_loss, but only from an evaluation monitor

    :param monitor: either a training or evaluation monitor
    :returns: the data for the given monitor directly
    """
    if monitor.is_evaluation:
        raise ValueError("Evaluation monitors do not have the total_loss key")
    else:
        return monitor.total_loss


def retrieve_constraint(monitor, absolute_value=False, original=True):
    """Retrieves the percentiles of the constraint (or the absolute value of
    the ith-constriant). WARNING: assumes only one constraint
    
    :param monitor: either a training or evaluation monitor
    :param index: index of the constraint
    :param absolute_value: whether to take the absolute value of the constraint
    :param original: whether to retreive the
    :returns: the percentiles of the ith-constraint
    """
    if absolute_value:
        object_string = "constraints_abs_percentiles"
    else:
        object_string = "constraints_percentiles"
    if monitor.is_evaluation:
        if original:
            true_object_string = f"original_{object_string}"
        else:
            true_object_string = f"final_{object_string}"
    else:
        true_object_string = object_string

    return getattr(monitor, true_object_string)


def retrieve_loss(monitor, absolute_value=False):
    """Retrieves the percentiles of the loss (or the absolute value of the loss)
    
    :param monitor: either a training or evaluation monitor
    :param absolute_value: whether to take the absolute value of the loss
    :returns: the percentiles of the loss
    """
    if absolute_value:
        return monitor.constraints_abs_percentiles
    else:
        return monitor.constraints_percentiles


def retrieve_parameters(monitor, gradients=False, differences=False):
    """Retrieves the parameters of the model or their gradients

    :param monitor: either a training or evaluation monitor
    :param gradients: whether to retrieve the gradients instead of the values
    :param differences: whether to retrieve the differences in parameters from
        original to final
    :returns: the values of the parameters (or gradients) for each batch and 
        epoch
    """
    if monitor.is_evaluation and differences:
        data = monitor.model_parameter_differences_percentiles
    elif not monitor.is_evaluation:
        if gradients:
            data = monitor.model_parameters_grad_percentiles
            if data[0][0] is None:
                raise AttributeError("Parameters do not have gradients")
        else:
            data = monitor.model_parameters_percentiles
    else:
        raise ValueError(
            "Monitor is evaluation, but differences was not requested"
        )

    return data

