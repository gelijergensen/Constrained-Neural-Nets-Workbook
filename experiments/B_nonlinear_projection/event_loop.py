"""The functions which perform the actual training and inference of a model,
given some possible configurations"""

from copy import deepcopy
from enum import Enum
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
import numpy as np
import torch

try:
    from time import perf_counter
except ImportError:
    from time import time as perf_counter

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
    PROJECTION_ITERATION = "projection_iteration"


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options."""
    xb, yb = batch
    return (
        tuple(
            convert_tensor(x, device=device, non_blocking=non_blocking)
            for x in xb
        ),
        convert_tensor(yb, device=device, non_blocking=non_blocking),
    )


def end_section(engine, section_event, section_start_time):
    """End the section, tabulate the time, fire the event, and resume time"""
    engine.state.times[section_event.value] = (
        perf_counter() - section_start_time
    )
    engine.fire_event(section_event)
    return perf_counter()


class TrainingLoop(object):
    @staticmethod
    def mean_squared_error(constraints):
        return torch.mean(constraints * constraints)

    def __init__(
        self,
        model,
        loss_fn,
        constraint_fn,
        optimizer,
        regularization_weight,
        error_fn,
        guard=True,
        device="cpu",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.constraint_fn = constraint_fn
        self.optimizer = optimizer
        self.regularization_weight = regularization_weight
        self.error_fn = (
            error_fn if error_fn is not None else self.mean_squared_error
        )
        self.guard = guard
        self.device = torch.device(device)

    def __call__(self, engine, batch):
        if not hasattr(engine.state, "times"):
            setattr(engine.state, "times", dict())

        iteration_start = perf_counter()
        section_start = iteration_start
        if self.optimizer is not None:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        engine.state.xb, engine.state.yb = prepare_batch(
            batch, device=self.device
        )
        section_start = end_section(
            engine, Sub_Batch_Events.DATA_LOADED, section_start
        )

        engine.state.out = self.model(*engine.state.xb)
        section_start = end_section(
            engine, Sub_Batch_Events.FORWARD_PASS_COMPLETED, section_start
        )

        if self.guard:
            # Ensure training isn't failing
            last = getattr(engine.state, "last", None)
            if (
                last is not None
                and len(engine.state.out) == len(last)
                and torch.allclose(engine.state.out, last)
            ):
                print("WARNING! Just outputting same thing!")
                print(f"xb: {[x.cpu() for x in engine.state.xb]}")
                print(f"yb: {engine.state.yb.cpu()}")
                print(f"out: {engine.state.out.cpu()}")
            engine.state.last = engine.state.out
            if torch.allclose(
                engine.state.out,
                engine.state.out.new_zeros(engine.state.out.size()),
            ):
                print("WARNING! Training is failing")
        section_start = end_section(
            engine, Sub_Batch_Events.GUARD_COMPLETED, section_start
        )

        engine.state.loss = self.loss_fn(engine.state.out, engine.state.yb)
        engine.state.mean_loss = torch.mean(engine.state.loss)
        section_start = end_section(
            engine, Sub_Batch_Events.LOSS_COMPUTED, section_start
        )

        engine.state.constraints, engine.state.constraints_diagnostics = self.constraint_fn(
            engine.state.out, engine.state.xb, self.model, True
        )  # last parameter is to return diagnostics
        engine.state.constraints_error = self.error_fn(engine.state.constraints)
        section_start = end_section(
            engine, Sub_Batch_Events.CONSTRAINTS_COMPUTED, section_start
        )

        engine.state.total_loss = (
            engine.state.mean_loss
            + self.regularization_weight * engine.state.constraints_error
        )
        section_start = end_section(
            engine, Sub_Batch_Events.REWEIGHTED_LOSS_COMPUTED, section_start
        )

        # log the values of the model parameters (without gradients)
        engine.state.model_parameters = (
            torch.cat(
                [param.view(-1) for param in self.model.parameters()], dim=-1
            )
            .clone()
            .detach()
        )
        if self.optimizer is not None:
            # backwards...
            engine.state.total_loss.backward()
            # attach the gradients
            engine.state.model_parameters_grad = torch.cat(
                [param.grad.view(-1) for param in self.model.parameters()],
                dim=-1,
            )
            # ...and step
            self.optimizer.step()
            engine.state.optimizer_state_dict = self.optimizer.state_dict()
            section_start = end_section(
                engine, Sub_Batch_Events.OPTIMIZER_STEPPED, section_start
            )
        else:
            engine.state.model_parameters_grad = None
            engine.state.optimizer_state_dict = None
        engine.state.model_state_dict = self.model.state_dict()

        engine.state.times["total"] = perf_counter() - iteration_start
        return engine.state.xb, engine.state.yb, engine.state.out


class ProjectionLoop(object):
    @staticmethod
    def mean_squared_error(constraints):
        return torch.mean(constraints * constraints)

    def __init__(
        self,
        model,
        loss_fn,
        constraint_fn,
        optimizer,
        regularization_weight,
        error_fn,
        device="cpu",
    ):
        self.model = model
        self.loss_fn = loss_fn  # we only use this for diagnostics
        self.constraint_fn = constraint_fn
        self.optimizer = optimizer
        self.regularization_weight = regularization_weight
        self.error_fn = (
            error_fn if error_fn is not None else self.mean_squared_error
        )
        self.device = torch.device(device)

    def __call__(self, engine, batch):
        # Used to restore model (both are important here)
        self.original_model_state_dict = deepcopy(self.model.state_dict())
        self.original_opt_state_dict = deepcopy(
            self.optimizer.state_dict().copy()
        )
        # Just used for diagnostics
        self.original_model_parameters = (
            torch.cat(
                [param.view(-1) for param in self.model.parameters()], dim=-1
            )
            .clone()
            .detach()
        )
        if not hasattr(engine.state, "times"):
            setattr(engine.state, "times", dict())

        iteration_start = perf_counter()
        section_start = iteration_start
        self.model.proj()  # Needs to be a ProjectableModel
        self.optimizer.zero_grad()
        engine.state.xb, engine.state.yb = prepare_batch(
            batch, device=self.device
        )

        section_start = end_section(
            engine, Sub_Batch_Events.DATA_LOADED, section_start
        )

        engine.state.out = self.model(*engine.state.xb)
        section_start = end_section(
            engine, Sub_Batch_Events.FORWARD_PASS_COMPLETED, section_start
        )

        engine.state.loss = self.loss_fn(engine.state.out, engine.state.yb)
        engine.state.mean_loss = torch.mean(engine.state.loss)
        section_start = end_section(
            engine, Sub_Batch_Events.LOSS_COMPUTED, section_start
        )

        engine.state.constraints, engine.state.constraints_diagnostics = self.constraint_fn(
            engine.state.out, engine.state.xb, self.model, True
        )  # last parameter is to return diagnostics
        engine.state.constraints_error = self.error_fn(engine.state.constraints)
        section_start = end_section(
            engine, Sub_Batch_Events.CONSTRAINTS_COMPUTED, section_start
        )

        engine.state.model_parameters = (
            torch.cat(
                [param.view(-1) for param in self.model.parameters()], dim=-1
            )
            .clone()
            .detach()
        )
        self.optimizer.zero_grad()
        engine.state.model_parameters_grad = torch.cat(
            [param.grad.view(-1) for param in self.model.parameters()], dim=-1
        )
        engine.state.constraints_error.backward()
        self.optimizer.step()
        engine.state.optimizer_state_dict = self.optimizer.state_dict()
        section_start = end_section(
            engine, Sub_Batch_Events.OPTIMIZER_STEPPED, section_start
        )
        engine.state.model_state_dict = self.model.state_dict()

        engine.state.times["total"] = perf_counter() - iteration_start
        return engine.state.xb, engine.state.yb, engine.state.out


def create_engine(
    model,
    loss_fn,
    constraint_fn,
    optimizer=None,
    projection=False,
    monitor=None,
    guard=True,
    regularization_weight=0.0,
    error_fn=None,
    device="cpu",
    tolerance=1e-5,
    max_iterations=1e4,
):
    """Creates an engine with the necessary components. If optimizer is not
    provided, then will run inference

    :param model: model to train or evaluate
    :param loss_fn: loss_fn to be used for training or monitored for evaluation
    :param constraint_fn: constraint function to be used for training or
        monitored for evaluation
    :param optimizer: optimizer to use to update the model. Must be provided 
        even for inference
    :param projection: whether to run the projection loop
    :param monitor: handler to be used for monitoring. Must have an
        .attach(engine) method
    :param guard: whether to perform a check to ensure that the model is
        training
    :param regularization_weight: multiplier to use for soft-constraining during
        training. Defaults to 0 for unconstrained
    :param error_fn: error function to use for converting the constraint 
        function to an error function for soft constraining. Defaults to MSE
    :param device: "cuda" or "cpu"
    :returns: an ignite.engine.Engine whose output is (xb, yb, out) for every
        iteration
    """

    if projection:
        iteration_fn = ProjectionLoop
    else:
        iteration_fn = TrainingLoop

    engine = Engine(
        iteration_fn(
            model,
            loss_fn,
            constraint_fn,
            optimizer,
            regularization_weight,
            error_fn,
            device,
        )
    )
    engine.register_events(*Sub_Batch_Events)

    if monitor is not None:
        monitor.attach(engine)

    return engine
