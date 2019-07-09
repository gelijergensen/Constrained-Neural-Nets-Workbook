import torch

from pyinsulate.derivatives import jacobian


def constrain_loss(loss, constraints, parameters, warn=True):
    """Reweights a given loss with the constraints by computing the optimal
    lagrange multipliers

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :returns: the new loss tensor after accounting for the constraints
    """
    return _constrain_loss(loss, constraints, parameters, warn=warn, allow_unused=True)


def _constrain_loss(loss, constraints, parameters, warn=True, allow_unused=False):
    """Assumes that the constraints are well-conditioned and computes the
    constrained loss

    :throws: RuntimeError if the jacobian of the constraints are not full rank
    """
    # The main function which we are computing is (loss + constraints . lambda)
    # lambda = $(J_g(x) J_g^T(x))^{-1}(g(x) - J_g(x) J_f^T(x)),
    #   where f is the loss and g is the constraints vector

    batched = loss.dim() > 0
    # Handle the special case of only one constraint
    if constraints.dim() == loss.dim():
        constraints = constraints.unsqueeze(-1)

    jac_f = torch.cat(
        jacobian(loss, parameters, batched=batched,
                 create_graph=True, allow_unused=allow_unused),
        dim=-1)
    jac_g = torch.cat(
        jacobian(constraints, parameters, batched=batched,
                 create_graph=True, allow_unused=allow_unused),
        dim=-1)
    if batched:
        jac_fT = jac_f.unsqueeze(-1)
        jacobian_metric = torch.einsum('bij,bkj->bik', jac_g, jac_g)
        try:
            inverse = jacobian_metric.inverse()
        except RuntimeError as rte:
            if warn:
                print("Error occurred while computing constrained loss:")
                print(rte)
                print("Constraints are likely ill-conditioned (i.e. jacobian is"
                      " not full rank at this point). Falling back to computing"
                      " pseudoinverse")
                if warn == "error":
                    raise rte
            inverse = jacobian_metric.pinverse()
        lambda_pre_inverse = torch.baddbmm(
            constraints.unsqueeze(-1), jac_g, jac_fT, alpha=-1)
        # batched version of g . INV * PRE_INV
        weighted_sum = torch.einsum(
            'bi,bij,bjk->b', constraints, inverse, lambda_pre_inverse)

    else:
        jac_fT = jac_f.transpose(0, 1)
        jacobian_metric = torch.einsum('ij,kj->ik', jac_g, jac_g)
        try:
            inverse = jacobian_metric.inverse()
        except RuntimeError as rte:
            if warn:
                print("Error occurred while computing constrained loss:")
                print(rte)
                print("Constraints are likely ill-conditioned (i.e. jacobian is"
                      " not full rank at this point). Falling back to computing"
                      " pseudoinverse")
                if warn == "error":
                    raise rte
            inverse = jacobian_metric.pinverse()
        lambda_pre_inverse = torch.addmm(
            constraints.unsqueeze(-1), jac_g, jac_fT, alpha=-1)
        # unbatched version of g . INV * PRE_INV
        weighted_sum = torch.einsum(
            'i,ij,jk->', constraints, inverse, lambda_pre_inverse)

    return loss + weighted_sum
