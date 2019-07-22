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

from pyinsulate.lagrange.exact import compute_multipliers

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
    return tuple(convert_tensor(x, device=device, non_blocking=non_blocking) for x in batch)


def create_engine(model, loss_fn, constraint_fn, optimizer=None, metrics=None, monitor=None, guard=True, **constraint_kwargs):
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
    :param constraint_kwargs: all other parameters will be passed along to the
        constraint function
    :returns: an ignite.engine.Engine whose output is (xb, yb, out) for every
        iteration
    """

    def end_section(engine, section_event, section_start_time):
        """End the section, tabulate the time, fire the event, and resume time"""
        engine.state.times[section_event.value] = perf_counter() - \
            section_start_time
        engine.fire_event(section_event)
        return perf_counter()

    def proof_of_constraint_iteration(engine, batch):
        if not hasattr(engine.state, 'times'):
            setattr(engine.state, 'times', dict())

        iteration_start = perf_counter()
        section_start = iteration_start
        if optimizer is not None:
            model.train()
            optimizer.zero_grad()
        engine.state.xb, engine.state.yb = prepare_batch(batch)

        section_start = end_section(
            engine, Sub_Batch_Events.DATA_LOADED, section_start)
        engine.state.out = model(engine.state.xb)

        section_start = end_section(
            engine, Sub_Batch_Events.FORWARD_PASS_COMPLETED, section_start)
        if guard:
            # Ensure training isn't failing
            last = getattr(engine.state, "last", None)
            if last is not None and len(engine.state.out) == len(last) and torch.allclose(engine.state.out, last):
                print("WARNING! Just outputting same thing!")
            engine.state.last = engine.state.out
            if torch.allclose(engine.state.out, engine.state.out.new_zeros(engine.state.out.size())):
                print("WARNING! Training is failing")

        section_start = end_section(
            engine, Sub_Batch_Events.GUARD_COMPLETED, section_start)
        engine.state.loss = loss_fn(engine.state.out, engine.state.yb)

        section_start = end_section(
            engine, Sub_Batch_Events.LOSS_COMPUTED, section_start)
        engine.state.constraints = constraint_fn(
            engine.state.out, engine.state.xb, **constraint_kwargs)

        section_start = end_section(
            engine, Sub_Batch_Events.CONSTRAINTS_COMPUTED, section_start)
        # TODO Eventually, we will want to upgrade this to be method specific
        engine.state.multipliers = compute_multipliers(
            engine.state.loss, engine.state.constraints, list(model.parameters()))
        # This computes either a batched or unbatched dot product
        # FIXME This only works when there is 1 constraint because the loss is
        # FIXME a single tensor and no longer batched
        engine.state.constrained_loss = engine.state.loss + \
            torch.einsum('...i,...i->...',
                         engine.state.constraints,
                         engine.state.multipliers
                         )

        section_start = end_section(
            engine, Sub_Batch_Events.REWEIGHTED_LOSS_COMPUTED, section_start)
        if optimizer is not None:
            engine.state.constrained_loss.backward()
            optimizer.step()
        engine.state.model_state_dict = model.state_dict()
        if optimizer is not None:
            engine.state.optimizer_state_dict = optimizer.state_dict()
        else:
            engine.state.optimizer_state_dict = None

        section_start = end_section(
            engine, Sub_Batch_Events.OPTIMIZER_STEPPED, section_start)

        engine.state.times['total'] = perf_counter() - iteration_start
        return engine.state.xb, engine.state.yb, engine.state.out

    engine = Engine(proof_of_constraint_iteration)
    engine.register_events(*Sub_Batch_Events)

    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(engine, name)

    if monitor is not None:
        monitor.attach(engine)

    return engine
