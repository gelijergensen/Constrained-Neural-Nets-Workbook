"""Functions for exactly computing the optimal Lagrange multipliers"""

from enum import Enum
import torch

try:
    from time import perf_counter
except ImportError:
    from time import time as perf_counter

from pyinsulate.derivatives import jacobian


class Timing_Events(Enum):
    """A set of Sub-Batch events"""

    COMPUTE_JF = "multipliers: compute loss jacobian"
    COMPUTE_JG = "multipliers: compute constraint jacobian"
    COMPUTE_GRAM = "multipliers: compute gram matrix"
    COMPUTE_PRE_MULTIPLIERS = "multipliers: compute pre-multipliers"
    CHOLESKY = "multipliers: cholesky"
    CHOLESKY_SOLVE = "multipliers: choleksy solve"
    ERRORED = "multipliers: errored"
    LEAST_SQUARES = "multipliers: least squares"


def compute_exact_multipliers(
    loss,
    constraints,
    parameters,
    return_timing=False,
    allow_unused=False,
    warn=True,
):
    """Assumes that the constraints are well-conditioned and computes the
    optimal Lagrange multipliers

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param return_timing: whether to also return the timing data
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :param allow_used: whether to allow some parameter to not be an input of
        the loss or constraints function. Defaults to False
    :returns: multipliers (, timing), if the timing is also requested. 
        Multipliers will have the same shape as the constraints
    :throws: RuntimeError if the jacobian of the constraints are not full rank
    """
    # multipliers = (J(g(s)) J(g(s))^T)^{-1} (g(s) - J(g(s)) J(f(s))^T)
    #   where f is the loss, g is the constraints vector, and s is the
    #   paramters of the neural network

    timing = dict()

    def record_timing(start_time, event):
        end_time = perf_counter()
        timing[event.value] = end_time - start_time
        return end_time

    # Restructure the constraints into the shapes (batchsize, -1)
    batchsize = 1 if len(loss.size()) == 0 else loss.size()[0]
    original_constraints_size = constraints.size()
    constraints = constraints.view(batchsize, -1)
    # Restructure the loss into the shape (batchsize, )
    loss = loss.view(batchsize)

    start_time = perf_counter()

    # Even though the loss is batched, the parameters are not, so we compute
    # the jacobian in an unbatched way and reassemble
    jac_fT = torch.cat(
        [
            jac.view(*loss.size(), -1)
            for jac in jacobian(
                loss,
                parameters,
                batched=False,
                create_graph=True,
                allow_unused=allow_unused,
            )
        ],
        dim=-1,
    )
    start_time = record_timing(start_time, Timing_Events.COMPUTE_JF)
    jac_g = torch.cat(
        [
            jac.view(*constraints.size(), -1)
            for jac in jacobian(
                constraints,
                parameters,
                batched=False,
                create_graph=True,
                allow_unused=allow_unused,
            )
        ],
        dim=-1,
    )
    start_time = record_timing(start_time, Timing_Events.COMPUTE_JG)

    # Possibly batched version of J(g) * J(g)^T
    gram_matrix = torch.einsum("...ij,...kj->...ik", jac_g, jac_g)
    start_time = record_timing(start_time, Timing_Events.COMPUTE_GRAM)

    untransformed_multipliers = (
        constraints - torch.einsum("...ij,...j->...i", jac_g, jac_fT)
    ).unsqueeze(-1)
    start_time = record_timing(
        start_time, Timing_Events.COMPUTE_PRE_MULTIPLIERS
    )

    try:
        # We do this this way because there is some chance we can provide this externally later
        cholesky_L = torch.cholesky(gram_matrix)
        start_time = record_timing(start_time, Timing_Events.CHOLESKY)
        multipliers = torch.cholesky_solve(
            untransformed_multipliers, cholesky_L
        )
        start_time = record_timing(start_time, Timing_Events.CHOLESKY_SOLVE)
        timing[Timing_Events.ERRORED.value] = False
        timing[Timing_Events.LEAST_SQUARES.value] = -999.0
    except RuntimeError as rte:
        if warn:
            print("Error occurred while computing constrained loss:")
            print(rte)
            print(
                "Constraints are likely ill-conditioned (i.e. jacobian is"
                " not full rank at this point)! Investingating..."
            )
            det_idx = torch.argmin(
                torch.stack(
                    [
                        torch.abs(torch.det(gram_matrix[i]))
                        for i in range(len(gram_matrix))
                    ]
                )
            )
            det = torch.det(gram_matrix[det_idx])
            print(f"Minimum abs(det(gram_matrix)): {det}")
            if torch.allclose(det, det.new_zeros(det.size())):
                print(
                    "Jacobian is indeed not full rank... Falling back to computing pseudoinverse"
                )
            else:
                print("Unknown reason for error. Printing complete diagnostics")
                print(f"jac_g.size(): {jac_g.size()}")
                print(
                    f"Jacobian of smallest abs(det(gram_matrix)): {jac_g[det_idx]}"
                )
                print(
                    f"gram_matrix of smallest abs(det(gram_matrix)): {gram_matrix[det_idx]}"
                )
                print("Falling back to computing pseudoinverse")

            if warn == "error":
                raise rte
        multipliers = untransformed_multipliers.new_zeros(
            untransformed_multipliers.size()
        )
        # torch.gels is NOT yet batch-enabled. As such, we do the batching manually
        for b in range(len(multipliers)):
            # discard the QR decomposition
            multipliers[b], __ = torch.gels(
                untransformed_multipliers[b], gram_matrix[b]
            )
        start_time = record_timing(start_time, Timing_Events.LEAST_SQUARES)
        timing[Timing_Events.ERRORED.value] = True
        timing[Timing_Events.CHOLESKY_SOLVE.value] = -999.0
        if Timing_Events.CHOLESKY.value not in timing:
            timing[Timing_Events.CHOLESKY.value] = -999.0

    if return_timing:
        return multipliers.view(original_constraints_size), timing
    else:
        return multipliers.view(original_constraints_size)
