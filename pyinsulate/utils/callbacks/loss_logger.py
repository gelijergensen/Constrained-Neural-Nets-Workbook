import numpy as np

from .callback import Callback


class LossLogger(Callback):
    """A Callback which logs the loss of a single model on the training or 
    validation data"""

    def __init__(self, train=False, valid=False, model_type=None):
        """Initializes the logger to operate on one or both of the datasets and
        the type of model

        :param train: whether to keep track of the loss on the training data
        :param valid: whether to keep track of the loss on the validation data
        :param model_type: string for the prefix of the model name on the loss.
            Defaults to None for a normal model, but should be 'gener' or 
            'discr' for a generator or discriminator
        """
        super().__init__()
        self.train = train
        self.valid = valid
        if model_type is None:
            self.loss_object_string = 'loss'
        else:
            self.loss_object_string = '%s_loss' % model_type

    def initialize(self, context):
        if self.train:
            self.train_losses = list()
        if self.valid:
            self.valid_losses = list()

    def on_train_epoch_start(self, context):
        if self.train:
            self.train_batchsizes = list()
            self.train_batchlosses = list()

    def on_train_batch_end(self, context):
        if self.train:
            self.train_batchsizes.append(context.get('batchsize'))
            self.train_batchlosses.append(
                context.get(self.loss_object_string).item()
            )

    def on_train_epoch_end(self, context):
        if self.train:
            self.train_losses.append(np.average(
                self.train_batchlosses, weights=self.train_batchsizes))

    def on_valid_epoch_start(self, context):
        if self.valid:
            self.valid_batchsizes = list()
            self.valid_batchlosses = list()

    def on_valid_batch_end(self, context):
        if self.valid:
            self.valid_batchsizes.append(context.get('batchsize'))
            self.valid_batchlosses.append(
                context.get(self.loss_object_string).item()
            )

    def on_valid_epoch_end(self, context):
        if self.valid:
            self.valid_losses.append(np.average(
                self.valid_batchlosses, weights=self.valid_batchsizes))
