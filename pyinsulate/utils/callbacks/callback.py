

class Callback(object):

    def initialize(self, context):
        pass

    def on_train_batch_start(self, context):
        pass

    def on_train_batch_end(self, context):
        pass

    def on_train_epoch_start(self, context):
        pass

    def on_train_epoch_end(self, context):
        pass

    def on_valid_batch_start(self, context):
        pass

    def on_valid_batch_end(self, context):
        pass

    def on_valid_epoch_start(self, context):
        pass

    def on_valid_epoch_end(self, context):
        pass
