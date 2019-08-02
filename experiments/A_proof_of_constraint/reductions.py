"""Reductions to apply to the constraints"""

import torch


class Lp_Reduction(object):
    """Computes the mean of the p-th elementwise power of the constraints along 
    the batch for each constraint"""

    def __init__(self, p):
        self.p = p
        self.even = p % 2 == 0

    def __call__(self, constraints):
        if self.p == 1:
            return torch.mean(torch.abs(constraints), dim=1)
        if self.even:
            return torch.mean(torch.power(constraints), dim=1)
        else:
            return torch.mean(torch.power(torch.abs(constraints)), dim=1)

    def __str__(self):
        return f"L{self.p}_Reduction()"


class Huber_Reduction(object):
    """Computes the Huber error along the batch for each constraint
    
    huber_error(a) = |a| > delta ? 0.5*a^2 : delta(|a| - 0.5*delta)
    """

    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, constraints):
        constraints_abs = torch.abs(constraints)
        return torch.mean(
            torch.where(
                constraints_abs > delta,
                0.5 * constraints_abs * constraints_abs,
                delta * (constraints_abs - 0.5 * delta),
            ),
            dim=1,
        )

    def __str__(self):
        return f"Huber_Reduction({self.delta})"

