"""Implementation of the wave PDE"""

import numpy as np
import torch

from src.derivatives import jacobian, jacobian_and_laplacian

__all__ = ["helmholtz_equation", "pythagorean_equation"]


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

    frequency = (2 * np.pi * parameterization[..., 1]).view(outputs.size())
    # r$ \nabla^2 u = - k^2 u$
    lhs = lap
    rhs = -frequency * frequency * outputs

    if return_diagnostics:
        return lhs - rhs, (lhs, rhs, jac)
    else:
        return lhs - rhs


def pythagorean_equation(
    outputs, inputs, parameterization, return_diagnostics=False
):
    """Computes the Pythagorean equation ((f * y)^2 + (y')^2 = f^2), assuming 
    that the network should satisfy that f * f * y = y'' (Helmholtz)
    
    :param outputs: output of some network
    :param inputs: inputs to some network
    :param parameterization: parameterization of the PDE which we expect to 
        follow: [amplitude, frequency, phase]
    :param return_diagnostics: whether to return an object containing 
        diagnostics information
    :return PDE value (, diagnostics tuple)
    """
    batched = len(inputs.size()) > 1
    jac = jacobian(
        outputs, inputs, batched=batched, create_graph=True, allow_unused=False
    )

    frequency = (2 * np.pi * parameterization[..., 1]).view(outputs.size())
    # r$ (f * y)^2 + (y')^2 = 1$
    lhs = (frequency * outputs) ** 2 + jac.view(outputs.size()) ** 2
    rhs = lhs.new_ones(lhs.size()) * (frequency ** 2)
    if return_diagnostics:
        return lhs - rhs, (lhs, rhs, jac)
    else:
        return lhs - rhs
