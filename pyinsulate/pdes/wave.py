"""Implementation of the wave PDE"""

import numpy as np
import torch

from pyinsulate.derivatives import jacobian_and_laplacian, divergence

__all__ = ["helmholtz_equation"]


def helmholtz_equation(
    outputs, inputs, parameterization, return_diagnostics=False
):
    """Computes the Helmholtz equation (time independent wave equation) value, 
    given the model inputs, outputs, and paramerization of the wave
    
    :param outputs: output of some network
    :param inputs: inputs to some network
    :param parameterization: parameterization of the PDE which we expect to 
        follow: [amplitude, frequency, phase]
    :param return_diagnostics: whether to return an object containing 
        diagnostics information
    :return PDE value (, diagnostics tuple)
    """
    batched = len(inputs.size()) > 1
    jac, lap = jacobian_and_laplacian(
        outputs, inputs, batched=batched, create_graph=True, allow_unused=False
    )

    frequency = 2 * np.pi * parameterization[..., 1]
    # r$ \nabla^2 u = - k^2 u$
    lhs = lap
    rhs = -(frequency * frequency).view(outputs.size()) * outputs
    if return_diagnostics:
        return lhs - rhs, (lhs, rhs, jac)
    else:
        return lhs - rhs
