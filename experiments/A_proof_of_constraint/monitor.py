"""Things to monitor during the training and evaluation phases"""

from pyinsulate.handlers.monitor import Monitor


class TrainingMonitor(Monitor):

    def __call__(self, engine):
        self.set('loss', engine.state.loss)
        self.set('constraints', engine.state.constraints)


class EvaluationMonitor(Monitor):

    def __call__(self, engine):
        self.set('metrics', engine.state.metrics)
