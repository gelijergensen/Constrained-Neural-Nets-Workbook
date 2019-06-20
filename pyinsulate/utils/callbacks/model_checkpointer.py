import os
import torch

from .callback import Callback


class Checkpointer(Callback):
    """A Callback which saves out the model after certain epochs"""

    def __init__(self, directory, base_filename, model_type=None, save_frequency=10):
        """Initializes the logger to operate on one or both of the datasets and
        the type of model

        :param directory: directory to save the checkpoints in
        :param base_filename: base string for the filenames. Epoch number will
            be appended
        :param model_type: string for the prefix of the model name on the loss.
            Defaults to None for a normal model, but should be 'gener' or 
            'discr' for a generator or discriminator
        :param save_frequency: number of epochs between model checkpoints
        """
        super().__init__()
        self.directory = directory
        self.base_filename = base_filename
        if model_type is None:
            self.model_type = "model"
        else:
            self.model_type = model_type
        self.save_frequency = save_frequency

    def initialize(self, context):
        try:
            os.makedirs(self.directory)
        except FileExistsError:
            pass  # directory already exists

    def on_train_epoch_end(self, context):
        epoch = context.get('epoch') + 1
        if epoch % self.save_frequency == 0:
            with open("%s/%s_%i.pkl" % (self.directory, self.base_filename, epoch), "wb") as f:
                torch.save(context.get(self.model_type), f)
