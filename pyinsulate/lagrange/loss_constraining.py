"""Convenience functions for reweighting the loss functions"""

import torch

from pyinsulate.lagrange.exact import compute_multipliers


def average_constrained_loss(
    loss,
    constraints,
    parameters,
    return_multipliers=False,
    warn=True,
    return_timing=False,
):
    """Computes the average constrained loss within the batch

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param return_multipliers: whether to also return the computed multipliers
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    """
    multipliers, timing = compute_batched_multipliers(
        loss, constraints, parameters, warn=warn, return_timing=True
    )
    multipliers = multipliers.detach()
    # mean of (loss + batched/unbatched dot product)
    constrained_loss = torch.mean(
        loss + torch.einsum("...i,...i->...", constraints, multipliers)
    )
    if return_multipliers:
        if return_timing:
            return constrained_loss, multipliers, timing
        else:
            return constrained_loss, multipliers
    else:
        if return_timing:
            return constrained_loss, timing
        else:
            return constrained_loss


def batchwise_constrained_loss(
    loss,
    constraints,
    parameters,
    return_multipliers=False,
    return_timing=False,
    warn=True,
):
    """Computes the average loss constrained by all constraints at once

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param return_multipliers: whether to also return the computed multipliers
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    """
    mean_loss = torch.mean(loss)
    multipliers, timing = compute_batchwise_multipliers(
        mean_loss, constraints, parameters, warn=warn, return_timing=True
    )
    multipliers = multipliers.detach()
    constrained_loss = mean_loss + torch.einsum(
        "i,i->", constraints.view(-1), multipliers
    )
    if return_multipliers:
        if return_timing:
            return constrained_loss, multipliers, timing
        else:
            return constrained_loss, multipliers
    else:
        if return_timing:
            return constrained_loss, timing
        else:
            return constrained_loss


def compute_batched_multipliers(
    loss, constraints, parameters, return_timing=False, warn=True
):
    """Computing the optimal Lagrange multipliers independently along the batch
    axis. CAUTION: these multipliers are differentiable, which may
    not be desired. In that case, .detach() the result

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param return_timing: whether to also return the timing data
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :returns: the optimal Lagrange multipliers as a single vector and maybe also
        the timing data
    """
    return compute_multipliers(
        loss,
        constraints,
        parameters,
        warn=warn,
        allow_unused=True,
        return_timing=return_timing,
    )


def compute_batchwise_multipliers(
    mean_loss, constraints, parameters, return_timing=False, warn=True
):
    """Computes the optimal Lagrange multipliers using all constraints within
    the batch at once. CAUTION: these multipliers are differentiable, which may
    not be desired. In that case, .detach() the result

    :param mean_loss: tensor corresponding to the mean of the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param return_timing: whether to also return the timing data
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :returns: the optimal Lagrange multipliers as a single vector and maybe also
        the timing data
    """
    return compute_multipliers(
        mean_loss,
        constraints.view(-1),
        parameters,
        warn=warn,
        return_timing=return_timing,
    )
