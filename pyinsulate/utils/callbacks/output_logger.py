import torch

from .callback import Callback


class OutputLogger(Callback):
    """A Callback which logs the output of a single model with regards to a 
    single training and/or validation example"""

    def __init__(self, train=False, valid=False, model_type='gener'):
        """Initializes the logger to operate on one or both of the datasets and
        the type of model

        :param train: whether to monitor on the training data
        :param valid: whether to monitor on the validation data
        :param model_type: string for the model type. Defaults to 'gener'.
            Should be None for a normal model
        """
        super().__init__()
        self.train = train
        self.valid = valid
        if model_type == 'gener':
            self.model_type = model_type
        else:
            raise NotImplementedError(
                '%s not yet supported in OutputLogger' % model_type)

    def initialize(self, context):
        device = next(context.get(self.model_type).parameters()).device
        # Grab the first item in the train_dl / valid_dl to monitor
        if self.train:
            self.train_xb, self.train_yb = next(iter(context.get('train_dl')))
            self.train_xb = self.train_xb[0].to(device)
            self.train_yb = self.train_yb[0].to(device)
            self.train_outputs = list()
        if self.valid:
            self.valid_xb, self.valid_yb = next(iter(context.get('valid_dl')))
            self.valid_xb = self.valid_xb[0].to(device)
            self.valid_yb = self.valid_yb[0].to(device)
            self.valid_outputs = list()

    def on_valid_epoch_end(self, context):
        if self.train:
            self.train_outputs.append(
                context.get(self.model_type)(
                    self.train_xb[None, ...])[0].cpu()
            )
        if self.valid:
            self.valid_outputs.append(
                context.get(self.model_type)(
                    self.valid_xb[None, ...])[0].cpu()
            )
