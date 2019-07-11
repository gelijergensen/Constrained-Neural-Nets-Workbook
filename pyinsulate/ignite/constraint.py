"""This class is copied wholesale from the Ignite codebase, with one important
change: under the hood, this uses our version of metric (GradientMetric), which
does not call torch.no_grad() on an update"""

from __future__ import division

from ignite.exceptions import NotComputableError
from pyinsulate.ignite.metric import GradientMetric


class GradientConstraint(GradientMetric):
    """
    Calculates the average constraint according to the passed constraint_fn.

    Args:
        constraint_fn (callable): a callable taking necessary input tensors and
            optionally other arguments, and returns the average constraint 
            tensor over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the constraint.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).

    """

    def __init__(self, constraint_fn, output_transform=lambda x: x,
                 batch_size=lambda x: x.shape[0]):
        super(GradientConstraint, self).__init__(output_transform)
        self._constraint_fn = constraint_fn
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        args = output[:-1]
        kwargs = output[-1]
        average_constraint = self._constraint_fn(*args, **kwargs)

        N = self._batch_size(args[0])
        self._sum += average_constraint * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'GradientConstraint must have at least one example before it can be computed.')
        return self._sum / self._num_examples
