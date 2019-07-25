"""Implementation of the wave PDE"""

import torch

from pyinsulate.derivatives import jacobian_and_laplacian, divergence

__all__ = ["helmholtz_equation"]


def helmholtz_equation(outputs, inputs, k=1):
    """Computes the Helmholtz equation (time independent wave equation) value, 
    given that the model has inputs x and outputs u(x)"""

    batched = len(inputs.size()) > 1

    jac, lap = jacobian_and_laplacian(
        outputs, inputs, batched=batched, create_graph=True, allow_unused=False
    )

    # r$ k^2 u + \nabla^2 u = 0 $
    return (k * k) * outputs + lap
