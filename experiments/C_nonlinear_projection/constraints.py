"""These are actually thin wrappers around the PDE constraints, but they 
correctly handle the passing of arguments"""

import numpy as np
from pyinsulate import pdes
import torch

__all__ = ["helmholtz_equation", "pythagorean_equation", "truth_residual"]


def helmholtz_equation(out, xb, model, return_diagnostics):
    return pdes.helmholtz_equation(out, *xb, return_diagnostics)


def pythagorean_equation(out, xb, model, return_diagnostics):
    return pdes.helmholtz_equation(out, *xb, return_diagnostics)


def truth_residual(out, xb, model, return_diagnostics):
    """The constraint here is the signed distance from the truth"""
    x, parameterization = xb

    amplitude = parameterization[..., 0].view(x.size())
    frequency = (2 * np.pi * parameterization[..., 1]).view(x.size())
    phase = parameterization[..., 2].view(x.size())

    truth = amplitude * torch.sin(frequency * x + phase)

    if return_diagnostics:
        return truth - out, (truth, out)
    else:
        return truth - out
