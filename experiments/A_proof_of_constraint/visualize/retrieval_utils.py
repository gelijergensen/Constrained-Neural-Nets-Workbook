"""Tools which are important for the retrieval of data to visualize, but aren't
part of the visualization process itself"""

import numpy as np
import torch


def retrieve_object(monitor, object_string, **kwargs):
    """Retrives the requested data from the given monitor

    :param monitor: either a training or evaluation monitor
    :param object_string: string to identify the object to retrieve. Should be
        one of "mean_loss", "constrained_loss", "reduced_constraints", 
        "constraints", "model_parameters"
    :param kwargs: all other options passed to downstream methods

    :returns: a numpy array of the data. If average_batches, then has shape
        (num_epochs,). Otherwise has shape (num_epochs, num_batchs_per_epoch)
    """
    if object_string in [
        "mean_loss",
        "constrained_loss",
        "reduced_constraints",
    ]:
        data = retrieve_plain(monitor, object_string)
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


def retrieve_plain(monitor, object_string):
    """Retrieves the request object as-is (doesn't apply any modification). This
    is valid only for objects which are single values (items from a tensor)

    :param monitor: either a training or evaluation monitor
    :param object_string: string to identify the object to retrieve
    :returns: the data for the given monitor directly
    """
    return getattr(monitor, object_string)


def retrieve_constraint(monitor, index=0, absolute_value=False):
    """Retrieves the percentiles of the ith-constraint (or the absolute value of
    the ith-constriant)
    
    :param monitor: either a training or evaluation monitor
    :param absolute_value: whether to take the absolute value of the constraint
    :returns: the percentiles of the ith-constraint
    """
    if absolute_value:
        return monitor.constraints_abs_percentiles
    else:
        return monitor.constraints_percentiles


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


# def retrieve_constraints_diagnostics(monitor, index=0):
#     """Retrieves the average value of index-th object from the constraints
#     diagnostics

#     :param monitor: either a training or evaluation monitor
#     :param index: the index in the constraints diagnostics to retrieve
#     :returns: the values of the constraints diagnostics for each batch and epoch
#     """
#     data = [
#         [batch[index] for batch in epoch]
#         for epoch in monitor.constraints_diagnostics
#     ]

#     return np.array(
#         [[torch.mean(batch).item() for batch in epoch] for epoch in data]
#     )


def retrieve_parameters(monitor, gradients=False):
    """Retrieves the parameters of the model or their gradients

    :param monitor: either a training or evaluation monitor
    :param gradients: whether to retrieve the gradients instead of the values
    :returns: the values of the parameters (or gradients) for each batch and 
        epoch
    """
    if gradients:
        data = monitor.model_parameters_grad
        if data[0][0] is None:
            raise AttributeError("Parameters do not have gradients")
    else:
        data = monitor.model_parameters

    return data

