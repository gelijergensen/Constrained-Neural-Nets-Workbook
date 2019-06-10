import numpy as np
import torch
import torch.nn as nn

__all__ = ["train_gan_pair"]


def train_gan_pair(gener, discr, train_dl, valid_dl, gener_opt, discr_opt, check_stop, discr_loss_fn=nn.BCELoss(), extra_gener_loss_fn=None, real_label=1, fake_label=0, log=print):
    """Trains a generator/discriminator pair until the desired criterion is met

    :param gener: model which generates examples from the training inputs
    :param discr: model which evaluates real or generated outputs
    :param train_dl: the dataloader which provides training batches
    :param valid_dl: the dataloader which provides validation batches
    :param gener_opt: optimizer for the generator
    :param discr_opt: optimizer for the discriminator
    :param check_stop: a function which determines when to stop training:
        (training_statistics) -> bool (False ends training)
    :param discr_loss_fn: loss function for the discriminator. Defaults to
        nn.BCELoss
    :param extra_gener_loss_fn: an optional extra loss function for the 
        generator
    :param real_label: expected value for real outputs. Defaults to 1
    :param fake_label: expected value for fake outputs. Defaults to 0
    :param log: a function for logging: (epoch, train_loss, valid_loss) -> void
        Defaults to the standard print function
    """
    training_statistics = {
        'epoch': 0,
        'gen_train_loss': None,
        'gen_valid_loss': None,
        'dis_train_loss': None,
        'dis_valid_loss': None,
        'real_train_score': None,
        'real_valid_score': None,
        'fake_train_score': None,
        'fake_valid_score': None,
    }
    while check_stop(training_statistics):
        # Train
        discr.train()
        gener.train()
        gen_train_losses, dis_train_losses, real_train_scores, \
            fake_train_scores, train_counts = zip(
                *(gan_batched_losses(
                    gener, discr, discr_loss_fn, extra_gener_loss_fn, xb, yb,
                    gener_opt, discr_opt, real_label=real_label,
                    fake_label=fake_label
                ) for xb, yb in train_dl)
            )
        # Validate
        with torch.no_grad():
            discr.eval()
            gener.eval()
            gen_valid_losses, dis_valid_losses, real_valid_scores, \
                fake_valid_scores, valid_counts = zip(
                    *(gan_batched_losses(
                        gener, discr, discr_loss_fn, extra_gener_loss_fn, xb,
                        yb, real_label=real_label, fake_label=fake_label
                    ) for xb, yb in valid_dl)
                )
        # Log statistics
        training_statistics['gen_train_loss'] = np.average(
            gen_train_losses, weights=train_counts)
        training_statistics['gen_valid_loss'] = np.average(
            gen_valid_losses, weights=valid_counts)
        training_statistics['dis_train_loss'] = np.average(
            dis_train_losses, weights=train_counts)
        training_statistics['dis_valid_loss'] = np.average(
            dis_valid_losses, weights=valid_counts)
        training_statistics['real_train_score'] = np.average(
            real_train_scores, weights=train_counts)
        training_statistics['real_valid_score'] = np.average(
            real_valid_scores, weights=valid_counts)
        training_statistics['fake_train_score'] = np.average(
            fake_train_scores, weights=train_counts)
        training_statistics['fake_valid_score'] = np.average(
            fake_valid_scores, weights=valid_counts)
        training_statistics['epoch'] += 1
        log(training_statistics)


def gan_batched_losses(gener, discr, discr_loss_fn, extra_gener_loss_fn, xb, yb, gener_opt=None, discr_opt=None, real_label=1, fake_label=0):
    """Computes the losses for a generator/discriminator pair on a batch 
    (xb, yb). Updates the models if their opts are not None

    :param gener: model which generates examples from the training inputs
    :param discr: model which evaluates real or generated outputs
    :param discr_loss_fn: loss function for the discriminator
    :param extra_gener_loss_fn: an optional extra loss function for the 
        generator
    :param xb: batch of inputs
    :param yb: batch of targets
    :param gener_opt: optional optimizer for the generator. If not provided, 
        then generator is not updated
    :param discr_opt: optional optimizer for the discriminator. If not provided, 
        then discriminator is not updated
    :param real_label: expected value for real outputs. Defaults to 1
    :param fake_label: expected value for fake outputs. Defaults to 0
    :returns: generator loss, discriminator loss, 
        mean discriminator score of reals, mean discriminator score of fakes,
        batchsize
    """
    # Evaluate discriminator with all real
    discr_real_out = discr(yb)
    discr_real_target = torch.full_like(discr_real_out, real_label)
    discr_real_loss = discr_loss_fn(discr_real_out, discr_real_target)
    # Evaluate discriminator with all fake
    gener_out = gener(xb)
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

    real_score = torch.mean(discr_fake_out)
    fake_score = torch.mean(discr_real_out)
    return (
        gener_loss.item(), discr_loss.item(), real_score.item(),
        fake_score.item(), len(xb)
    )
