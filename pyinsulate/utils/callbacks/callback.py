

class Callback(object):

    def initialize(self, context):
        """Called whenever training is started for the first time"""
        pass

    def resume(self, context):
        """Called whenever training is resumed at a higher epoch than 0"""
        pass

    def pause(self, context):
        """Called at the end of training"""
        pass

    def on_train_batch_start(self, context):
        """Called whenever a training batch starts"""
        pass

    def on_train_batch_end(self, context):
        """Called whenever a training batch ends"""
        pass

    def on_train_epoch_start(self, context):
        """Called whenever a training epoch starts"""
        pass

    def on_train_epoch_end(self, context):
        """Called whenever a training epoch ends"""
        pass

    def on_valid_batch_start(self, context):
        """Called whenever a validation batch starts"""
        pass

    def on_valid_batch_end(self, context):
        """Called whenever a validation batch ends"""
        pass

    def on_valid_epoch_start(self, context):
        """Called whenever a validation epoch starts"""
        pass

    def on_valid_epoch_end(self, context):
        """Called whenever a validation epoch ends"""
        pass
