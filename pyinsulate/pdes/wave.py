"""Implementation of the wave PDE"""

import numpy as np
import torch

from pyinsulate.derivatives import jacobian_and_laplacian, divergence

__all__ = ["helmholtz_equation"]


def helmholtz_equation(outputs, inputs, parameterization):
    """Computes the Helmholtz equation (time independent wave equation) value, 
    given the model inputs, outputs, and paramerization of the wave"""
    batched = len(inputs.size()) > 1
    jac, lap = jacobian_and_laplacian(
        outputs, inputs, batched=batched, create_graph=True, allow_unused=False
    )

    frequency = 2 * np.pi * parameterization[..., 1]
    # r$ k^2 u + \nabla^2 u = 0 $
    return (frequency * frequency).view(outputs.size()) * outputs + lap
