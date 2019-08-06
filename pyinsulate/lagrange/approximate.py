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

    APPROXIMATE_JF = "multipliers: approximate loss jacobian"
    APPROXIMATE_JG = "multipliers: approximate constraint jacobian"
    COMPUTE_JF = "multipliers: compute loss jacobian"
    COMPUTE_JG = "multipliers: compute constraint jacobian"
    COMPUTE_GRAM = "multipliers: compute gram matrix"
    COMPUTE_PRE_MULTIPLIERS = "multipliers: compute pre-multipliers"
    CHOLESKY = "multipliers: cholesky"
    CHOLESKY_SOLVE = "multipliers: choleksy solve"
    ERRORED = "multipliers: errored"
    LEAST_SQUARES = "multipliers: least squares"
    RECOMPUTED_JACOBIANS = "multipliers: recomputed exact jacobians"
    UPDATE_STATE = "multipliers: update state"


class Jacobian_Approximation_State(object):
    """Stores the various objects necessary for appoximating the jacobians"""

    def __init__(
        self,
        loss_jacobian,
        constraint_jacobian,
        last_parameters,
        last_loss,
        last_constraints,
    ):
        # Technically, this is a vector
        self.loss_jacobian = loss_jacobian
        self.constraint_jacobian = constraint_jacobian
        self.last_parameters = last_parameters
        self.last_loss = last_loss
        self.last_constraints = last_constraints


def compute_approximate_multipliers(
    loss,
    constraints,
    parameters,
    state=None,
    return_timing=False,
    allow_unused=False,
    warn=True,
):
    """Assumes that the constraints are well-conditioned and approximates the
    optimal Lagrange multipliers by applying Broyden's trick to approximate the 
    jacobians. This requires that the constraints and loss be effectively only
    functions of the parameters, not the model input (i.e. a reduction must 
    have been applied along the batch axis of the loss and constraints)

    :param loss: tensor corresponding to the evalutated loss
    :param constraints: a single tensor corresponding to the evaluated
        constraints (you may need to torch.stack() first)
    :param parameters: an iterable of the parameters to optimize
    :param state: state of the approximation, as returned by this function. Set
        to None to reinitialize the function
    :param return_timing: whether to also return the timing data
    :param warn: whether to warn if the constraints are ill-conditioned. If set
        to "error", then will throw a RuntimeError if this occurs
    :param allow_used: whether to allow some parameter to not be an input of
        the loss or constraints function. Defaults to False
    :returns: multipliers, state (, timing), if the timing is also requested. 
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

    # Handle the special case of only one constraint
    original_constraints_size = constraints.size()
    if constraints.dim() == loss.dim():
        constraints = constraints.unsqueeze(-1)

    start_time = perf_counter()

    if state is None:
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
        state = Jacobian_Approximation_State(
            jac_fT,
            jac_g,
            [x.clone().detach() for x in parameters],
            loss.clone().detach(),
            constraints.clone().detach(),
        )
        start_time = record_timing(start_time, Timing_Events.UPDATE_STATE)
        timing[Timing_Events.APPROXIMATE_JF.value] = -999.0
        timing[Timing_Events.APPROXIMATE_JG.value] = -999.0
        timing[Timing_Events.RECOMPUTED_JACOBIANS.value] = True
    else:
        # Broyden's trick: (for iterative approximations to the Jacobian of the function y(s))
        # J(y_{k+1})~= Y_{k+1} = Y_k + (1/((s_{k+1} - s_k)^T(s_{k+1} - s_k)) ...
        #  * ((y_{k+1} - y_k) - Y_k (s_{k+1} - s_k)) (s_{k+1} - s_k)^T
        delta_parameters = torch.cat(
            [
                (new - old).view(-1)
                for (old, new) in zip(state.last_parameters, parameters)
            ]
        )
        delta_parameters_dot_product = delta_parameters @ delta_parameters
        delta_loss = (loss - state.last_loss).view(1)
        jac_fT = (
            state.loss_jacobian
            + torch.ger(
                delta_loss - (state.loss_jacobian @ delta_parameters),
                delta_parameters,
            )
            / delta_parameters_dot_product
        ).view(-1)
        start_time = record_timing(start_time, Timing_Events.APPROXIMATE_JF)
        delta_constraints = constraints - state.last_constraints
        jac_g = (
            state.constraint_jacobian
            + torch.ger(
                delta_constraints
                - (state.constraint_jacobian @ delta_parameters),
                delta_parameters,
            )
            / delta_parameters_dot_product
        )
        start_time = record_timing(start_time, Timing_Events.APPROXIMATE_JG)
        state = Jacobian_Approximation_State(
            jac_fT,
            jac_g,
            [x.clone().detach() for x in parameters],
            loss.clone().detach(),
            constraints.clone().detach(),
        )
        start_time = record_timing(start_time, Timing_Events.UPDATE_STATE)
        timing[Timing_Events.COMPUTE_JF.value] = -999.0
        timing[Timing_Events.COMPUTE_JG.value] = -999.0
        timing[Timing_Events.RECOMPUTED_JACOBIANS.value] = False

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
                " not full rank at this point). Falling back to computing"
                " pseudoinverse"
            )
            if warn == "error":
                raise rte

        multipliers = untransformed_multipliers.new_zeros(
            untransformed_multipliers.size()
        )
        if len(original_constraints_size) > 1:
            # torch.gels is NOT yet batch-enabled. As such, we do the batching manually
            for b in range(len(multipliers)):
                # discard the QR decomposition
                multipliers[b], __ = torch.gels(
                    untransformed_multipliers[b], gram_matrix[b]
                )
        else:
            # not batched
            multipliers, __ = torch.gels(untransformed_multipliers, gram_matrix)
        start_time = record_timing(start_time, Timing_Events.LEAST_SQUARES)
        timing[Timing_Events.ERRORED.value] = True
        timing[Timing_Events.CHOLESKY_SOLVE.value] = -999.0
        if Timing_Events.CHOLESKY.value not in timing:
            timing[Timing_Events.CHOLESKY.value] = -999.0

    if return_timing:
        return multipliers.view(original_constraints_size), state, timing
    else:
        return multipliers.view(original_constraints_size), state

