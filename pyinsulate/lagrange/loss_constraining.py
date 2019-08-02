"""Convenience functions for reweighting the loss functions"""

import torch

from pyinsulate.lagrange.exact import compute_exact_multipliers

__all__ = ["constrain_loss"]


def constrain_loss(
    loss,
    constraints,
    parameters,
    batchwise=False,
    reduction=None,
    return_multipliers=False,
    return_timing=False,
    warn=True,
):
    """Computes the lagrange multipliers according to some particular batching
    method with a possible reduction

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param batchwise: whether to treat all instances of the constraints across
        the batch as separate constraints. If set to True, will ignore the 
        "reduction" argument. WARNING: batchwise constraining is unstable 
        because the constraints are not linearly independent!
    :param reduction: a function which takes the constraints tensor of shape 
        (batch_size, num_constraints) and returns a reduced tensor of shape
        (num_constraints). Typically this is a constraint-wise "error" function.
        If batchwise is set to True, this argument is ignored
    :param return_multipliers: whether to also return the computed multipliers
    :param return_timing: whether to also return the timing data
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :returns: constrained_loss (, multipliers) (, timing), if the multipliers
        and/or timing are also requested. constrained_loss will be a tensor of
        shape (batch_size,) only if reduction=None and batchwise=False. 
        Otherwise, it will have shape (1,) 
    """
    if batchwise:
        reduced_loss = torch.mean(loss)
        reduced_constraints = constraints.view(-1)
    elif reduction is not None:
        reduced_loss = torch.mean(loss)
        reduced_constraints = reduction(constraints)
    else:
        reduced_loss = loss
        reduced_constraints = constraints

    multipliers, timing = compute_exact_multipliers(
        reduced_loss,
        reduced_constraints,
        parameters,
        warn=warn,
        allow_unused=True,
        return_timing=True,
    )

    # We don't want to back-prop through the multipliers themselves
    multipliers = multipliers.detach()
    constrained_loss = reduced_loss + torch.einsum(
        "...i,...i->...", reduced_constraints, multipliers
    )  # possibly batched dot product

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
