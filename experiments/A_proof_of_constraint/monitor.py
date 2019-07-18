"""Things to monitor during the training and evaluation phases"""

from pyinsulate.handlers.monitor import Monitor

from .event_loop import Sub_Batch_Events


class ProofOfConstraintMonitor(Monitor):

    def __init__(self):
        super().__init__()

        self.add('epoch', average=True)
        self.add('loss', average=False)
        self.add('batch_size', average=False)
        self.add('constraints', average=False)
        self.time_keys = ['total'] + [event.value for event in Sub_Batch_Events]
        for key in self.time_keys:
            self.add(key, average=True)

    def __call__(self, engine):
        self.set('epoch', engine.state.epoch)
        self.set('loss', engine.state.loss.item())
        self.set('batch_size', len(engine.state.xb))
        self.set('constraints', engine.state.constraints.to('cpu'))
        for key in self.time_keys:
            self.set(key, engine.state.times[key])
