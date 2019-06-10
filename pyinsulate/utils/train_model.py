import numpy as np
import torch

__all__ = ["train_model"]


def train_model(model, train_dl, valid_dl, loss_fn, opt, check_stop, log=print):
    """Trains a model until the desired criterion is met

    :param model: a model to train
    :param train_dl: the dataloader which provides training batches
    :param valid_dl: the dataloader which provides validation batches
    :param loss_fn: a loss function: (model_output, target) -> number
    :param opt: optimizer
    :param check_stop: a function which determines when to stop training:
        (training_statistics) -> bool (False ends training)
    :param log: a function for logging: (training_statistics) -> void
        Defaults to the standard print function
    """
    training_statistics = {
        'epoch': 0,
        'train_loss': None,
        'valid_loss': None,
    }
    while check_stop(training_statistics):
        # Train
        model.train()
        train_losses, train_counts = zip(
            *(batched_loss(model, loss_fn, xb, yb, opt) for xb, yb in train_dl)
        )
        # Validate
        model.eval()
        with torch.no_grad():
            valid_losses, valid_counts = zip(
                *(batched_loss(model, loss_fn, xb, yb) for xb, yb in valid_dl)
            )
        # Log statistics
        training_statistics['train_loss'] = np.average(
            train_losses, weights=train_counts)
        training_statistics['valid_loss'] = np.average(
            valid_losses, weights=valid_counts)
        training_statistics['epoch'] += 1
        log(training_statistics)


def batched_loss(model, loss_fn, xb, yb, opt=None):
    """Computes the loss for a model on a batch (xb, yb). Updates the model if
    opt is not None

    :param model: a model to compute the loss for
    :param loss_fn: a loss function: (model_output, target) -> number
    :param xb: batch of inputs
    :param yb: batch of targets
    :param opt: optional optimizer. If not provided, then model is not updated
    :returns: loss, batchsize
    """
    loss = loss_fn(model(xb), yb)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(xb)
