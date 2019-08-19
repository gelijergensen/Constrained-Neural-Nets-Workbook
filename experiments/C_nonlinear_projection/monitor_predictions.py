"""This logger keeps track of the model predictions during training and 
projecting"""

from ignite.engine import Events
import numpy as np
import torch


class PredictionLogger(object):
    def __init__(self, model):
        self.inputs = None
        self.outputs = None
        self.model = model
        self.predictions = list()

    def __call__(self, engine):
        dataset = engine.state.dataloader.dataset
        x_list = list()
        param_list = list()
        y_list = list()
        for i in range(len(dataset)):
            batch = dataset[i]
            x_list.append(batch[0][0])
            param_list.append(batch[0][1])
            y_list.append(batch[1])
        xs = torch.stack(x_list)
        params = torch.stack(param_list)
        if self.inputs is None:
            self.inputs = xs.detach().numpy()
        if self.outputs is None:
            self.outputs = torch.stack(y_list).detach().numpy()
        self.model.eval()
        preds = self.model(xs, params).detach().numpy()
        self.predictions[-1].append(preds)

    def new_epoch(self, engine):
        self.predictions.append(list())

    def attach(self, trainer, projector=None):
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.new_epoch)
        if projector is not None:
            projector.add_event_handler(Events.STARTED, self.__call__)
            projector.add_event_handler(Events.EPOCH_COMPLETED, self.__call__)

