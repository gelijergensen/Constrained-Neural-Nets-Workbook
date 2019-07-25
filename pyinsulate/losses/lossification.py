"""These are tools which will convert an entire tensor into a single scalar
tensor, allowing for it to be a loss term"""

import functools

import torch


def lossify(lossifier):
    """A decorator which converts a tensor-output to a loss function by applying
    the given lossifier function"""

    def conversion_decorator(tens_func):
        @functools.wraps(tens_func)
        def lossified_func(*args, **kwargs):
            return lossifier(tens_func(*args, **kwargs))

        return lossified_func

    return conversion_decorator


def mean_of_sum_of_squares(tensor):
    """For batched inputs, computes the mean (along the batch) of the sum of the
    squares of the values of a tensor"""
    return torch.mean(
        torch.sum(tensor * tensor, dim=list(range(1, len(tensor.size()))))
    )
