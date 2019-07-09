"""Tools to wrap around losses and ensure that they are given the correct
parameters"""

import functools


def takes_xb_yb(loss_fn):
    """A decorator which converts a loss function taking (xb, yb) to an 
    equivalent one accepting (xb, yb, ypred)"""
    @functools.wraps(loss_fn)
    def transformed_loss_fn(xb, yb, ypred):
        return loss_fn(xb, yb)
    return transformed_loss_fn


def takes_xb_ypred(loss_fn):
    """A decorator which converts a loss function taking (xb, ypred) to an 
    equivalent one accepting (xb, yb, ypred)"""
    @functools.wraps(loss_fn)
    def transformed_loss_fn(xb, yb, ypred):
        return loss_fn(xb, ypred)
    return transformed_loss_fn


def takes_yb_ypred(loss_fn):
    """A decorator which converts a loss function taking (yb, ypred) to an
    equivalent one accepting (xb, yb, ypred)"""
    @functools.wraps(loss_fn)
    def transformed_loss_fn(xb, yb, ypred):
        return loss_fn(yb, ypred)
    return transformed_loss_fn
