"""Tools for differentiably computing derivatives"""

import torch
from torch import autograd


__all__ = [
    "jacobian",
    "jacobian_and_hessian",
    "trace",
    "divergence",
    "jacobian_and_laplacian",
]


def _get_size(tensor):
    """Returns the size of a tensor, but treats a scalar as a 1-element vector"""
    if tensor.dim() == 0:
        return torch.Size([1])
    else:
        return tensor.size()


def _jacobian(y, xs, create_graph, allow_unused):
    """Computes the jacobian of outputs with respect to inputs

    :param y: tensor for the output of some function
    :param xs: a list of tensors for the inputs of some function
    :param create_graph: set True for the resulting jacobian to be differentible
    :param allow_unused: set False to assert all inputs affected the outputs
    :returns: a list of tensors of size (y.size() + xs[i].size())
        containing the jacobians of outputs with respect to inputs
    """

    jacs = [
        y.new_zeros((*_get_size(y), *_get_size(x))).view(-1, *_get_size(x))
        for x in xs
    ]
    flat_y = y.view(-1)
    for i in range(flat_y.size()[-1]):
        cols_i = autograd.grad(
            flat_y[i],
            xs,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )

        for j, col_i in enumerate(cols_i):
            if col_i is None:
                # this element doesn't depend on the xs, so leave gradient 0
                continue
            else:
                jacs[j][i] = col_i

    for j in range(len(jacs)):
        if create_graph:
            jacs[j].requires_grad_()
        jacs[j] = jacs[j].view(*_get_size(y), *_get_size(xs[j]))

    return jacs


def _batched_jacobian(y, xs, create_graph, allow_unused):
    """Computes the jacobian of a vector batched outputs with respected to inputs

    :param y: tensor for the output of some vector function.
        size: (batchsize, *outsize)
    :param xs: list of tensors for the inputs of some vector function.
        size: (batchsize, *insize[i]) for each element of inputs
    :param create_graph: set True for the resulting jacobian to be differentible
    :param allow_unused: set False to assert all inputs affected the outputs
    :returns: a list of tensors of size (batchsize, *outsize, *insize[i])
        containing the jacobian of outputs with respect to inputs for batch
        element b in row b
    """
    batchsize = _get_size(y)[0]
    outsize = _get_size(y)[1:]
    query_vector = y.new_ones(batchsize)

    jacs = [
        y.new_zeros((batchsize, *outsize, *_get_size(x)[1:])).view(
            batchsize, -1, *_get_size(x)[1:]
        )
        for x in xs
    ]
    flat_y = y.reshape(batchsize, -1)
    for i in range(flat_y.size()[1]):
        cols_i = autograd.grad(
            flat_y[:, i],
            xs,
            grad_outputs=query_vector,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )
        for j, col_i in enumerate(cols_i):
            if col_i is None:
                # this element doesn't depend on the inputs, so leave gradient 0
                continue
            else:
                jacs[j][:, i] = col_i

    for j in range(len(jacs)):
        if create_graph:
            jacs[j].requires_grad_()
        jacs[j] = jacs[j].view(batchsize, *outsize, *_get_size(xs[j])[1:])

    return jacs


def jacobian(y, xs, batched=False, create_graph=False, allow_unused=False):
    """Computes the jacobian of y with respect to in

    :param y: output of some tensor function
    :param xs: either a single tensor input to some tensor function or a list
        of tensor inputs to a function
    :param batched: whether the first dimension of y and xs is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :param allow_unused: whether terms in xs are allowed to not contribute to
        any of y
    :returns: jacobian(s) of the same shape as xs. Each jacobian will have 
        size (*y.size(), *xs[i].size()) or 
        (batchsize, *y.size()[1:], *xs[i].size()[1:])
    """
    if isinstance(xs, list) or isinstance(xs, tuple):
        if batched:
            return _batched_jacobian(
                y, xs, create_graph=create_graph, allow_unused=allow_unused
            )
        else:
            return _jacobian(
                y, xs, create_graph=create_graph, allow_unused=allow_unused
            )
    else:
        xs_list = [xs]
        if batched:
            return _batched_jacobian(
                y, xs_list, create_graph=create_graph, allow_unused=allow_unused
            )[0]
        else:
            return _jacobian(
                y, xs_list, create_graph=create_graph, allow_unused=allow_unused
            )[0]


def jacobian_and_hessian(
    y, xs, batched=False, create_graph=False, allow_unused=False
):
    """Computes the jacobian and the hessian of y with respect to xs

    :param y: output of some tensor function
    :param xs: either a single tensor input to some tensor function or a list
        of tensor inputs to a function
    :param batched: whether the first dimension of y and xs is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :param allow_unused: whether terms in xs are allowed to not contribute to
        any of y
    :returns jacobian, hessian, which are lists if xs is a list
    """
    jac = jacobian(
        y, xs, batched=batched, create_graph=True, allow_unused=allow_unused
    )
    if isinstance(jac, list):
        hes = [
            jacobian(
                jac_i,
                xs,
                batched=batched,
                create_graph=create_graph,
                allow_unused=allow_unused,
            )
            for jac_i in jac
        ]
    else:
        hes = jacobian(
            jac,
            xs,
            batched=batched,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )
    return jac, hes


def trace(tensor):
    """Computes the trace of the last two dimensions of a tensor

    :param tensor: either a single tensor or a list of tensors
    :returns: either a single tensor of a list of tensors containing the trace 
        of the given tensor(s), with each of size (*tensor[i].size()[:-2], 1)
    """
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return [torch.einsum("...ii->...", t) for t in tensor]
    else:
        return torch.einsum("...ii->...", tensor)


def divergence(
    y, xs, jacobian=None, batched=False, create_graph=False, allow_unused=False
):
    """Computes the divergence of y with respect to xs. If jacobian is 
    provided, then will be more efficient

    :param y: output of some tensor function
    :param xs: input to some tensor function
    :param jacobian: optional parameter for the jacobian of out w.r.t. in
    :param batched: whether the first dimension of y and xs is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param allow_unused: whether terms in xs are allowed to not contribute to
        any of y
    :returns: divergence tensor. Size (1,) or (out.size()[0], 1)
    """
    if jacobian is None:
        jacobian = jacobian(
            y,
            xs,
            batched=batched,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )
    return trace(jacobian)


def jacobian_and_laplacian(
    y, xs, batched=False, create_graph=False, allow_unused=False
):
    """This currently computes the laplacian by using the entire hessian. There
    may be a more efficient way to do this

    :param y: output of some tensor function
    :param xs: input to some tensor function
    :param batched: whether the first dimension of y and xs is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :returns jacobian, laplacian
    """
    jac, hes = jacobian_and_hessian(
        y,
        xs,
        batched=batched,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )
    lap = trace(hes)
    return jac, lap
