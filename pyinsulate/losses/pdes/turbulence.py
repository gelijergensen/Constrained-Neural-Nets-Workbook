"""Implementation of particular PDEs"""

import torch

from pyinsulate.derivatives import jacobian_and_laplacian, divergence

__all__ = ["steady_state_turbulence"]


def steady_state_turbulence(outputs, inputs, nu=0.01):
    """Computes the steady-state turbulence PDE magnitude, given that the model
    has inputs x,y,z and outputs v_x, v_y, v_z"""

    batched = len(inputs.size()) > 1

    jac, lap = jacobian_and_laplacian(
        outputs, inputs, batched=batched, create_graph=True, allow_unused=False)
    # r$ \nabla \cdot u = 0 $
    div = divergence(outputs, inputs, jacobian=jac, batched=batched)

    lhs = torch.einsum('...j,...jk->...k', outputs, jac)
    rhs = nu * lap
    # r$ u \cdot \nabla u - \nu \nabla^2 u = 0 $
    sst = lhs - rhs
    return torch.cat([div.unsqueeze(-1), sst], dim=-1)
