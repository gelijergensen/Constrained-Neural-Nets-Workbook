"""Tools which are important for the retrieval of data to visualize, but aren't
part of the visualization process itself"""

import numpy as np
import torch


def retrieve_object(monitor, object_string, **kwargs):
    """Retrives the requested data from the given monitor

    :param monitor: either a training or evaluation monitor
    :param object_string: string to identify the object to retrieve. Should be
        one of 
    :param kwargs: all other options passed to downstream methods

    :returns: the data for the given monitor in a plottable format (a numpy
        array of single values)
    """
    if object_string in [
        "mean_loss",
        "constrained_loss",
        "reduced_constraints",
    ]:
        return retrieve_plain(monitor, object_string)
    elif object_string == "constraints":
        return retrieve_singlevalue_constraints(monitor, **kwargs)
    elif object_string == "constraints_diagnostics":
        return retrieve_constraints_diagnostics(monitor, **kwargs)
    else:
        raise ValueError(f"{object_string} not recognized for data retrieval")


def retrieve_plain(monitor, object_string):
    """Retrieves the request object as-is (doesn't apply any modification). This
    is valid only for objects which are single values (items from a tensor)

    :param monitor: either a training or evaluation monitor
    :param object_string: string to identify the object to retrieve
    :returns: the data for the given monitor in a plottable format (a numpy
        array of single values)
    """
    data = monitor.get(object_string)
    batch_sizes = monitor.batch_size

    return batch_weighted_average(data, batch_sizes)


def retrieve_singlevalue_constraints(monitor, absolute_value=False):
    """Retrieves the average value (or magnitude thereof) of the constraints.
    Warning: this method assumes that the constraint is a tensor of shape 
    (batch_size, ) for each batch
    
    :param monitor: either a training or evaluation monitor
    :param absolute_value: whether to take the absolute value of the constraints
    :returns: the data for the given monitor in a plottable format (a numpy
        array of single values)
    """
    data = monitor.constraints
    batch_sizes = monitor.batch_size

    if absolute_value:
        batchwise_averages = np.array(
            [
                [torch.abs(torch.mean(batch)).item() for batch in epoch]
                for epoch in data
            ]
        )
    else:
        batchwise_averages = np.array(
            [[torch.mean(batch).item() for batch in epoch] for epoch in data]
        )

    return batch_weighted_average(batchwise_averages, batch_sizes)


def retrieve_constraints_diagnostics(monitor, index=0):
    """Retrieves the average value of index-th object from the constraints 
    diagnostics

    :param monitor: either a training or evaluation monitor
    :param index: the index in the constraints diagnostics to retrieve
    :returns: the data for the given monitor in a plottable format (a numpy
        array of single values)
    """
    data = [
        [batch[index] for batch in epoch]
        for epoch in monitor.constraints_diagnostics
    ]
    batch_sizes = monitor.batch_size

    values = np.array(
        [[torch.mean(batch).item() for batch in epoch] for epoch in data]
    )
    return batch_weighted_average(values, batch_sizes)


def batch_weighted_average(data, batch_sizes):
    """Returns the weighted average for each epoch of the data, where data and
    batch_sizes are both lists of lists, with the outer list corresponding to
    epochs and the inner list corresponding to iterations/batches"""
    return np.array(
        [
            np.average(datum, weights=batch_size)
            for datum, batch_size in zip(data, batch_sizes)
        ]
    )

