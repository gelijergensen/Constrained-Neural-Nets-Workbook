import numpy as np
import torch
import torch.nn as nn

__all__ = ["train_gan_pair"]


def train_gan_pair(gener, discr, train_dl, valid_dl, gener_opt, discr_opt, should_stop, discr_loss_fn=nn.BCELoss(), extra_gener_loss_fn=None, real_label=1, fake_label=0, log=print, callbacks=None, start_epoch=0):
    """Trains a generator/discriminator pair until the desired criterion is met

    :param gener: model which generates examples from the training inputs
    :param discr: model which evaluates real or generated outputs
    :param train_dl: the dataloader which provides training batches
    :param valid_dl: the dataloader which provides validation batches
    :param gener_opt: optimizer for the generator
    :param discr_opt: optimizer for the discriminator
    :param should_stop: a function which determines when to stop training:
        (context) -> bool (True ends training)
    :param discr_loss_fn: loss function for the discriminator. Defaults to
        nn.BCELoss
    :param extra_gener_loss_fn: an optional extra loss function for the 
        generator
    :param real_label: expected value for real outputs. Defaults to 1
    :param fake_label: expected value for fake outputs. Defaults to 0
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
        discr.train()
        gener.train()
        for xb, yb in train_dl:
            gan_training_loop(
                gener, discr, discr_loss_fn, extra_gener_loss_fn, xb, yb,
                real_label, fake_label, log, callbacks, False, gener_opt, discr_opt,
            )
        # Handle callbacks
        for callback in callbacks:
            callback.on_train_epoch_end(locals())
        # Handle callbacks
        for callback in callbacks:
            callback.on_valid_epoch_start(locals())
        # Validate
        with torch.no_grad():
            discr.eval()
            gener.eval()
            for xb, yb in valid_dl:
                gan_training_loop(
                    gener, discr, discr_loss_fn, extra_gener_loss_fn, xb, yb,
                    real_label, fake_label, log, callbacks, True
                )
        # Handle callbacks
        for callback in callbacks:
            callback.on_valid_epoch_end(locals())
        epoch += 1

    for callback in callbacks:
        callback.pause(locals())


def gan_training_loop(gener, discr, discr_loss_fn, extra_gener_loss_fn, xb, yb, real_label, fake_label, log, callbacks, is_validation, gener_opt=None, discr_opt=None):
    """Performs a forwards and possibly backwards pass of the GANs. Updates the
    models only if their respective opts are not None

    :param gener: model which generates examples from the training inputs
    :param discr: model which evaluates real or generated outputs
    :param discr_loss_fn: loss function for the discriminator
    :param extra_gener_loss_fn: an optional extra loss function for the 
        generator
    :param xb: batch of inputs
    :param yb: batch of targets
    :param real_label: expected value for real outputs
    :param fake_label: expected value for fake outputs
    :param is_validation: boolean for whether this is validation data
    :param gener_opt: optional optimizer for the generator. If not provided, 
        then generator is not updated
    :param discr_opt: optional optimizer for the discriminator. If not provided, 
        then discriminator is not updated
    :param log: a function for logging. Defaults to the standard print function
    :param callbacks: an optional list of callbacks to apply
    """
    # Handle callbacks
    if is_validation:
        for callback in callbacks:
            callback.on_valid_batch_start(locals())
    else:
        for callback in callbacks:
            callback.on_train_batch_start(locals())
    # Move xb and yb
    gener_device = next(gener.parameters()).device
    discr_device = next(discr.parameters()).device
    xb = xb.to(gener_device)
    yb = yb.to(discr_device)
    # Evaluate discriminator with all real
    discr_real_out = discr(yb)
    discr_real_target = torch.full_like(discr_real_out, real_label)
    discr_real_loss = discr_loss_fn(discr_real_out, discr_real_target)
    # Evaluate discriminator with all fake
    gener_out = gener(xb.to(gener_device)).to(discr_device)
    discr_fake_out = discr(gener_out)
    discr_fake_target = torch.full_like(discr_fake_out, fake_label)
    discr_fake_loss = discr_loss_fn(discr_fake_out, discr_fake_target)
    # Collect
    discr_loss = discr_real_loss + discr_fake_loss
    # Update discriminator
    if discr_opt is not None:
        discr_opt.zero_grad()
        discr_loss.backward(retain_graph=True)
        discr_opt.step()
    # Evaluate generator using discriminator before update
    gener_target = torch.full_like(discr_fake_out, real_label)
    if extra_gener_loss_fn is not None:
        gener_loss = discr_loss_fn(discr_fake_out, gener_target) + \
            extra_gener_loss_fn(gener_out, yb)
    else:
        gener_loss = discr_loss_fn(discr_fake_out, gener_target)
    # Update generator
    if gener_opt is not None:
        gener_opt.zero_grad()
        gener_loss.backward()
        gener_opt.step()
    batchsize = len(xb)
    # Handle callbacks
    if is_validation:
        for callback in callbacks:
            callback.on_valid_batch_end(locals())
    else:
        for callback in callbacks:
            callback.on_train_batch_end(locals())
