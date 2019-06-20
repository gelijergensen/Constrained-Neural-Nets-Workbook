import numpy as np
import torch

__all__ = ["train_model"]


def train_model(model, train_dl, valid_dl, loss_fn, opt, should_stop, log=print, callbacks=None, start_epoch=0):
    """Trains a model until the desired criterion is met

    :param model: a model to train
    :param train_dl: the dataloader which provides training batches
    :param valid_dl: the dataloader which provides validation batches
    :param loss_fn: a loss function: (model_output, target) -> number
    :param opt: optimizer
    :param should_stop: a function which determines when to stop training:
        (context) -> bool (True ends training)
    :param log: a function for logging. Defaults to the standard print function
    :param callbacks: an optional list of callbacks to apply
    :param start_epoch: epoch to start training. Set to a higher value than one
        to "resume" training
    """
    if callbacks is None:
        callbacks = list()

    epoch = start_epoch

    if epoch > 0:
        for callback in callbacks:
            callback.resume(locals())
    else:
        for callback in callbacks:
            callback.initialize(locals())

    while not should_stop(locals()):
        # Handle callbacks
        for callback in callbacks:
            callback.on_train_epoch_start(locals())
        # Train
        model.train()
        for xb, yb in train_dl:
            training_loop(model, loss_fn, xb, yb, log, callbacks, False, opt)
        # Handle callbacks
        for callback in callbacks:
            callback.on_train_epoch_end(locals())
        # Handle callbacks
        for callback in callbacks:
            callback.on_valid_epoch_start(locals())
        # Validate
        with torch.no_grad():
            model.eval()
            for xb, yb in valid_dl:
                training_loop(model, loss_fn, xb, yb, log, callbacks, True)
        # Handle callbacks
        for callback in callbacks:
            callback.on_valid_epoch_end(locals())
        epoch += 1

    for callback in callbacks:
        callback.pause(locals())


def training_loop(model, loss_fn, xb, yb, log, callbacks, is_validation, opt=None):
    """Computes the loss for a model on a batch (xb, yb). Updates the model if
    opt is not None

    :param model: a model to compute the loss for
    :param loss_fn: a loss function: (model_output, target) -> number
    :param xb: batch of inputs
    :param yb: batch of targets
    :param log: a function for logging. Defaults to the standard print function
    :param callbacks: an optional list of callbacks to apply
    :param is_validation: boolean for whether this is validation data
    :param opt: optional optimizer. If not provided, then model is not updated
    """
    # Handle callbacks
    if is_validation:
        for callback in callbacks:
            callback.on_valid_batch_start(locals())
    else:
        for callback in callbacks:
            callback.on_train_batch_start(locals())
    # Move xb and yb
    device = next(model.parameters()).device
    xb = xb.to(device)
    yb = yb.to(device)
    # Evaluate model
    loss = loss_fn(model(xb), yb)
    # Update model
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    batchsize = len(xb)
    # Handle callbacks
    if is_validation:
        for callback in callbacks:
            callback.on_valid_batch_end(locals())
    else:
        for callback in callbacks:
            callback.on_train_batch_end(locals())
