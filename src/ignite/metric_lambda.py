"""This class is copied wholesale from the Ignite codebase, with one important
change: under the hood, this uses our version of metric (GradientMetric), which
does not call torch.no_grad() on an update"""

from src.ignite.metric import GradientMetric
from ignite.engine import Events
import itertools


class GradientMetricsLambda(GradientMetric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.

    When update, this metric does not recursively update the metrics
    it depends on. When reset, all its dependency metrics would be
    resetted. When attach, all its dependencies would be automatically
    attached.

    Args:
        f (callable): the function that defines the computation
        args (sequence): Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.

    Example:

    .. code-block:: python

        precision = Precision(average=False)
        recall = Recall(average=False)

        def Fbeta(r, p, beta):
            return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

        F1 = GradientMetricsLambda(Fbeta, recall, precision, 1)
        F2 = GradientMetricsLambda(Fbeta, recall, precision, 2)
        F3 = GradientMetricsLambda(Fbeta, recall, precision, 3)
        F4 = GradientMetricsLambda(Fbeta, recall, precision, 4)
    """

    def __init__(self, f, *args, **kwargs):
        self.function = f
        self.args = args
        self.kwargs = kwargs
        super(GradientMetricsLambda, self).__init__()

    def reset(self):
        for i in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(i, GradientMetric):
                i.reset()

    def update(self, output):
        # NB: this method does not recursively update dependency metrics,
        # which might cause duplicate update issue. To update this metric,
        # users should manually update its dependencies.
        pass

    def compute(self):
        materialized = [
            i.compute() if isinstance(i, GradientMetric) else i
            for i in self.args
        ]
        materialized_kwargs = {
            k: (v.compute() if isinstance(v, GradientMetric) else v)
            for k, v in self.kwargs.items()
        }
        return self.function(*materialized, **materialized_kwargs)

    def _internal_attach(self, engine):
        for index, metric in enumerate(
            itertools.chain(self.args, self.kwargs.values())
        ):
            if isinstance(metric, GradientMetricsLambda):
                metric._internal_attach(engine)
            elif isinstance(metric, GradientMetric):
                if not engine.has_event_handler(
                    metric.started, Events.EPOCH_STARTED
                ):
                    engine.add_event_handler(
                        Events.EPOCH_STARTED, metric.started
                    )
                if not engine.has_event_handler(
                    metric.iteration_completed, Events.ITERATION_COMPLETED
                ):
                    engine.add_event_handler(
                        Events.ITERATION_COMPLETED, metric.iteration_completed
                    )

    def attach(self, engine, name):
        # recursively attach all its dependencies
        self._internal_attach(engine)
        # attach only handler on EPOCH_COMPLETED
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)
