import numpy as np
import torch


__all__ = ["train_model"]


def train_model(model, train_dl, valid_dl, loss_fn, opt, until, log=print):
    """Trains a model until the desired criterion is met

    :param model: a model to train
    :param train_dl: the dataloader which provides training batches
    :param valid_dl: the dataloader which provides validation batches
    :param loss_fn: a loss function: (model_output, target) -> number
    :param opt: optimizer
    :param until: a function which determines when to stop training:
        (epoch, loss) -> bool (False ends training)
    :param log: a function for logging: (epoch, train_loss, valid_loss) -> void
        Defaults to the standard print function
    """
    epoch = 0
    train_loss = None
    valid_loss = None
    while until(epoch, train_loss, valid_loss):
        model.train()
        train_losses, train_counts = zip(
            *(batched_loss(model, loss_fn, xb, yb, opt) for xb, yb in train_dl)
        )

        model.eval()
        with torch.no_grad():
            valid_losses, valid_counts = zip(
                *(batched_loss(model, loss_fn, xb, yb) for xb, yb in valid_dl)
            )

        train_loss = np.average(train_losses, weights=train_counts)
        valid_loss = np.average(valid_losses, weights=valid_counts)

        log(epoch, train_loss, valid_loss)
        epoch += 1


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
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
