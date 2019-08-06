"""The functions which perform the actual training and inference of a model,
given some possible configurations"""


from enum import Enum
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
import torch

try:
    from time import perf_counter
except ImportError:
    from time import time as perf_counter

from pyinsulate.lagrange import constrain_loss

__all__ = ["create_engine", "Sub_Batch_Events"]


class Sub_Batch_Events(Enum):
    """A set of Sub-Batch events"""

    DATA_LOADED = "load_data"
    FORWARD_PASS_COMPLETED = "forward_pass"
    GUARD_COMPLETED = "guard"
    LOSS_COMPUTED = "compute_loss"
    CONSTRAINTS_COMPUTED = "compute_constraints"
    REWEIGHTED_LOSS_COMPUTED = "compute_reweighted_loss"
    OPTIMIZER_STEPPED = "step_optimizer"


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options."""
    return tuple(
        convert_tensor(x, device=device, non_blocking=non_blocking)
        for x in batch
    )


def create_engine(
    model,
    loss_fn,
    constraint_fn,
    optimizer=None,
    metrics=None,
    monitor=None,
    guard=True,
    method="unconstrained",
    reduction=None,
    ground_approximation=None,
    **constraint_kwargs,
):
    """Creates an engine with the necessary components. If optimizer is not
    provided, then will run inference

    :param model: model to train or evaluate
    :param loss_fn: loss_fn to be used for training or monitored for evaluation
    :param constraint_fn: constraint function to be used for training or
        monitored for evaluation
    :param optimizer: optimizer to use to update the model. If not provided,
        then the model weights are not updated
    :param metrics: an optional dictionary of (ignite / pyinsulate.ignite)
        metrics to attach to the engine
    :param monitor: handler to be used for monitoring. Must have an
        .attach(engine) method
    :param guard: whether to perform a check to ensure that the model is
        training
    :param method: method to use for constraining. Should be one of
        "constrained" - compute average (along batch) of constrained update
        "batchwise" - compute constrained update of mean loss with respect to
            all constraints within the batch
        "reduction" - apply reduction before computing constraints. If no 
            reduction is specified, will throw error
        "approximate" - approximate the optimal constraints using Broyden's 
            trick. If no reduction is specified, will throw error
        "unconstrained" - don't constrain. Used as a control method
        "soft-constrained" - use soft constraints. If no reduction is specified,
            will throw error
        "no-loss" - intended entirely for debugging. Ignores the loss function
            entirely and just tries to satisfy the constraints
        "non-projecting" - the sum of "no-loss" and "unconstrained". This 
            destroys the exponential convergence guarantee, but should be useful
            for debugging
    :param reduction: reduction to apply to constraints before computing 
        constrained loss if method == "reduction", "approximate", or 
        "soft-constrained"
    :param ground_approximation: string for when to recompute the exact 
        multipliers to ground the approximation. Should consist of a number 
        followed by either "batches" or "epochs". e.g. "1 epochs", "3 batches"
    :param constraint_kwargs: all other parameters will be passed along to the
        constraint function
    :returns: an ignite.engine.Engine whose output is (xb, yb, out) for every
        iteration
    """
    # Setup whether we would ground on batches or epochs
    ground_epochs = (
        ground_approximation is not None and "epoch" in ground_approximation
    )
    if ground_approximation is not None:
        ground_frequency = int(ground_approximation.split(" ")[0])
    else:
        ground_frequency = None

    def should_ground(engine):
        # If this is evaluation, there's no way the parameters can change
        if optimizer is None:
            return True

        # We always ground on the first iteration ever
        if engine.state.last_grounded == 0:
            if ground_epochs:
                engine.state.last_grounded = engine.state.epoch
            else:
                engine.state.last_grounded = engine.state.iteration
            return True

        if ground_approximation is None:
            return False

        if ground_epochs:
            if (
                engine.state.epoch - engine.state.last_grounded
                == ground_frequency
            ):
                engine.state.last_grounded = engine.state.epoch
                return True
            else:
                return False
        else:
            if (
                engine.state.iteration - engine.state.last_grounded
                == ground_frequency
            ):
                engine.state.last_grounded = engine.state.iteration
                return True
            else:
                return False

    def end_section(engine, section_event, section_start_time):
        """End the section, tabulate the time, fire the event, and resume time"""
        engine.state.times[section_event.value] = (
            perf_counter() - section_start_time
        )
        engine.fire_event(section_event)
        return perf_counter()

    def proof_of_constraint_iteration(engine, batch):
        if not hasattr(engine.state, "last_grounded"):
            engine.state.last_grounded = 0
        if not hasattr(engine.state, "times"):
            setattr(engine.state, "times", dict())

        iteration_start = perf_counter()
        section_start = iteration_start
        if optimizer is not None:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()
        engine.state.xb, engine.state.yb = prepare_batch(batch)
        section_start = end_section(
            engine, Sub_Batch_Events.DATA_LOADED, section_start
        )

        engine.state.out = model(engine.state.xb)
        section_start = end_section(
            engine, Sub_Batch_Events.FORWARD_PASS_COMPLETED, section_start
        )

        if guard:
            # Ensure training isn't failing
            last = getattr(engine.state, "last", None)
            if (
                last is not None
                and len(engine.state.out) == len(last)
                and torch.allclose(engine.state.out, last)
            ):
                print("WARNING! Just outputting same thing!")
                print(f"xb: {engine.state.xb}")
                print(f"yb: {engine.state.yb}")
                print(f"out: {engine.state.out}")
            engine.state.last = engine.state.out
            if torch.allclose(
                engine.state.out,
                engine.state.out.new_zeros(engine.state.out.size()),
            ):
                print("WARNING! Training is failing")
        section_start = end_section(
            engine, Sub_Batch_Events.GUARD_COMPLETED, section_start
        )

        engine.state.loss = loss_fn(engine.state.out, engine.state.yb)
        engine.state.mean_loss = torch.mean(engine.state.loss)
        section_start = end_section(
            engine, Sub_Batch_Events.LOSS_COMPUTED, section_start
        )

        engine.state.constraints = constraint_fn(
            engine.state.out, engine.state.xb, **constraint_kwargs
        )
        section_start = end_section(
            engine, Sub_Batch_Events.CONSTRAINTS_COMPUTED, section_start
        )

        if method == "constrained":
            constrained_loss, engine.state.multipliers, multiplier_computation_timing = constrain_loss(
                engine.state.loss,
                engine.state.constraints,
                list(model.parameters()),
                return_multipliers=True,
                return_timing=True,
                # defaults are for this method
            )
            engine.state.constrained_loss = torch.mean(constrained_loss)
            engine.state.times.update(multiplier_computation_timing)
        elif method == "batchwise":
            engine.state.constrained_loss, engine.state.multipliers, multiplier_computation_timing = constrain_loss(
                engine.state.loss,
                engine.state.constraints,
                list(model.parameters()),
                return_multipliers=True,
                return_timing=True,
                batchwise=True,
            )
            engine.state.times.update(multiplier_computation_timing)
        elif method == "reduction":
            if reduction is None:
                raise ValueError(
                    "Reduction must be specified if method=='reduction'"
                )
            engine.state.constrained_loss, engine.state.multipliers, multiplier_computation_timing = constrain_loss(
                engine.state.loss,
                engine.state.constraints,
                list(model.parameters()),
                return_multipliers=True,
                return_timing=True,
                reduction=reduction,
            )
            engine.state.times.update(multiplier_computation_timing)
        elif method == "approximate":
            if reduction is None:
                raise ValueError(
                    "Reduction must be specified if method=='approximate'"
                )
            if not hasattr(engine.state, "approximation_state"):
                engine.state.approximation_state = None
            engine.state.constrained_loss, engine.state.approximation_state, engine.state.multipliers, multiplier_computation_timing = constrain_loss(
                engine.state.loss,
                engine.state.constraints,
                list(model.parameters()),
                approximate=True,
                state=(
                    None
                    if should_ground(engine)
                    else engine.state.approximation_state
                ),
                return_multipliers=True,
                return_timing=True,
                reduction=reduction,
            )
            engine.state.times.update(multiplier_computation_timing)
        elif method == "soft-constrained":
            if reduction is None:
                raise ValueError(
                    "Reduction must be specified if method=='soft-constrained'"
                )
            # Technically the multipliers are defined this way, so we set this for consistency
            reduced_constraints = reduction(engine.state.constraints)
            multiplier = engine.state.iteration / (
                100000 + engine.state.iteration
            )
            engine.state.multipliers = reduced_constraints.new_full(
                engine.state.constraints.size(), multiplier
            )
            engine.state.constrained_loss = torch.mean(
                engine.state.loss
            ) + multiplier * torch.sum(reduced_constraints)
        elif method == "unconstrained":
            # Technically the multipliers are zero, so we set this for consistency
            engine.state.multipliers = engine.state.constraints.new_zeros(
                engine.state.constraints.size()
            )
            engine.state.constrained_loss = torch.mean(engine.state.loss)
        elif method == "no-loss":
            constrained_loss, engine.state.multipliers, multiplier_computation_timing = constrain_loss(
                engine.state.loss.new_zeros(
                    engine.state.loss.size()
                ).requires_grad_(),
                engine.state.constraints,
                list(model.parameters()),
                return_multipliers=True,
                return_timing=True,
            )
            engine.state.constrained_loss = torch.mean(constrained_loss)
            engine.state.times.update(multiplier_computation_timing)
        elif method == "non-projecting":
            correction_term, engine.state.multipliers, multiplier_computation_timing = constrain_loss(
                engine.state.loss.new_zeros(
                    engine.state.loss.size()
                ).requires_grad_(),
                engine.state.constraints,
                list(model.parameters()),
                return_multipliers=True,
                return_timing=True,
            )
            engine.state.constrained_loss = torch.mean(
                engine.state.loss + correction_term
            )
            engine.state.times.update(multiplier_computation_timing)
        else:
            raise ValueError(f"Method {method} not known. Please respecify")

        section_start = end_section(
            engine, Sub_Batch_Events.REWEIGHTED_LOSS_COMPUTED, section_start
        )

        if optimizer is not None:
            engine.state.constrained_loss.backward()
            optimizer.step()
        engine.state.model_state_dict = model.state_dict()
        if optimizer is not None:
            engine.state.optimizer_state_dict = optimizer.state_dict()
        else:
            engine.state.optimizer_state_dict = None
        section_start = end_section(
            engine, Sub_Batch_Events.OPTIMIZER_STEPPED, section_start
        )

        if torch.allclose(
            engine.state.constrained_loss,
            engine.state.constrained_loss.new_zeros(
                engine.state.constrained_loss.size()
            ),
        ):
            print("Constrained loss is zero!")

        engine.state.times["total"] = perf_counter() - iteration_start
        return engine.state.xb, engine.state.yb, engine.state.out

    engine = Engine(proof_of_constraint_iteration)
    engine.register_events(*Sub_Batch_Events)

    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(engine, name)

    if monitor is not None:
        monitor.attach(engine)

    return engine
