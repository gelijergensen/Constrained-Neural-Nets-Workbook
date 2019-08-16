"""These are actually thin wrappers around the PDE constraints, but they 
correctly handle the passing of arguments"""

from pyinsulate import pdes

__all__ = ["helmholtz_equation", "pythagorean_equation"]


def helmholtz_equation(out, xb, model, return_diagnostics):
    return pdes.helmholtz_equation(out, *xb, return_diagnostics)


def pythagorean_equation(out, xb, model, return_diagnostics):
    return pdes.helmholtz_equation(out, *xb, return_diagnostics)

