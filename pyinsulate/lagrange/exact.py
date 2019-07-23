"""Functions for exactly computing the optimal Lagrange multipliers"""

import torch

from pyinsulate.derivatives import jacobian


def compute_multipliers(loss, constraints, parameters, warn=True):
    """Computing the optimal Lagrange multipliers

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :returns: the optimal Lagrange multipliers in the same shape as constraints
    """
    return _compute_multipliers(loss, constraints, parameters, warn=warn, allow_unused=True)


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
