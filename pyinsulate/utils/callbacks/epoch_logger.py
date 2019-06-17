from .callback import Callback


class EpochLogger(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = None

    def on_train_epoch_end(self, context):
        self.epoch = context.get('epoch')
