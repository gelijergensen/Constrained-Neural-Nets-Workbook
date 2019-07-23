"""Functions for exactly computing the optimal Lagrange multipliers"""

import torch

from pyinsulate.derivatives import jacobian


def average_constrained_loss(loss, constraints, parameters, return_multipliers=False, warn=True):
    """Computes the average constrained loss within the batch

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param return_multipliers: whether to also return the computed multipliers
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    """
    multipliers = compute_batched_multipliers(
        loss, constraints, parameters, warn=warn)
    # mean of (loss + batched/unbatched dot product)
    constrained_loss = torch.mean(
        loss + torch.einsum('...i,...i->...', constraints, multipliers)
    )
    if return_multipliers:
        return constrained_loss, multipliers
    else:
        return constrained_loss


def batchwise_constrained_loss(loss, constraints, parameters, return_multipliers=False, warn=True):
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
    multipliers = compute_batchwise_multipliers(
        mean_loss, constraints, parameters, warn=warn)
    constrained_loss = mean_loss + \
        torch.einsum('i,i->', constraints.view(-1), multipliers)
    if return_multipliers:
        return constrained_loss, multipliers
    else:
        return constrained_loss


def compute_batched_multipliers(loss, constraints, parameters, warn=True):
    """Computing the optimal Lagrange multipliers independently along the batch
    axis

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :returns: the optimal Lagrange multipliers in the same shape as constraints
    """
    return _compute_multipliers(loss, constraints, parameters, warn=warn, allow_unused=True)


def compute_batchwise_multipliers(mean_loss, constraints, parameters, warn=True):
    """Computes the optimal Lagrange multipliers using all constraints within
    the batch at once

    :param mean_loss: tensor corresponding to the mean of the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :returns: the optimal Lagrange multipliers as a single vector
    """
    return _compute_multipliers(mean_loss, constraints.view(-1), parameters, warn=warn)


def _compute_multipliers(loss, constraints, parameters, warn=True, allow_unused=False):
    """Assumes that the constraints are well-conditioned and computes the
    optimal Lagrange multipliers

    :throws: RuntimeError if the jacobian of the constraints are not full rank
    """
    # The main function which we are computing is (loss + constraints . weights)
    # lambda = $(J_g(x) J_g^T(x))^{-1}(g(x) - J_g(x) J_f^T(x)),
    #   where f is the loss and g is the constraints vector

    batched = loss.dim() > 0
    # Handle the special case of only one constraint
    original_constraints_size = constraints.size()
    if constraints.dim() == loss.dim():
        constraints = constraints.unsqueeze(-1)

    # Even though the loss is batched, the parameters are not, so we compute
    # the jacobian in an unbatched way and reassemble
    jac_f = torch.cat([jac.view(*loss.size(), -1) for jac in
                       jacobian(loss, parameters, batched=False,
                                create_graph=True, allow_unused=allow_unused)],
                      dim=-1)
    jac_g = torch.cat([jac.view(*constraints.size(), -1) for jac in
                       jacobian(constraints, parameters, batched=False,
                                create_graph=True, allow_unused=allow_unused)],
                      dim=-1)
    jac_fT = jac_f.unsqueeze(-1)
    if batched:
        # gram matrix of constraint jacobian
        gram_matrix = torch.einsum('bij,bkj->bik', jac_g, jac_g)
        try:
            gram_inverse = gram_matrix.inverse()
        except RuntimeError as rte:
            if warn:
                print("Error occurred while computing constrained loss:")
                print(rte)
                print("Constraints are likely ill-conditioned (i.e. jacobian is"
                      " not full rank at this point). Falling back to computing"
                      " pseudoinverse")
                if warn == "error":
                    raise rte
            gram_inverse = gram_matrix.pinverse()
        untransformed_weights = torch.baddbmm(
            constraints.unsqueeze(-1), jac_g, jac_fT, alpha=-1)
        # batched version of INV * PRE_INV
        multipliers = torch.einsum(
            'bij,bjk->bi', gram_inverse, untransformed_weights)

    else:
        gram_matrix = torch.einsum('ij,kj->ik', jac_g, jac_g)
        try:
            gram_inverse = gram_matrix.inverse()
        except RuntimeError as rte:
            if warn:
                print("Error occurred while computing constrained loss:")
                print(rte)
                print("Constraints are likely ill-conditioned (i.e. jacobian is"
                      " not full rank at this point). Falling back to computing"
                      " pseudoinverse")
                if warn == "error":
                    raise rte
            gram_inverse = gram_matrix.pinverse()
        untransformed_weights = torch.addmm(
            constraints.unsqueeze(-1), jac_g, jac_fT, alpha=-1)

        # unbatched version of INV * PRE_INV
        multipliers = torch.einsum(
            'ij,jk->i', gram_inverse, untransformed_weights)

    return multipliers.view(original_constraints_size)
