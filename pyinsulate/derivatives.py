import torch
from torch import autograd


__all__ = ["jacobian", "jacobian_and_hessian", "trace", "divergence",
           "jacobian_and_laplacian"]


def _jacobian(outputs, inputs, create_graph, allow_unused):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :param allow_unused: set False to assert all inputs affected the outputs
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=allow_unused)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def _batched_jacobian(outputs, inputs, create_graph, allow_unused):
    """Computes the jacobian of a vector batched outputs with respected to inputs

    :param outputs: matrix for the output of some vector function.
        size: (batchsize, *outsize)
    :param inputs: matrix for the input of some vector function.
        size: (batchsize, *insize)
    :param create_graph: set True for the resulting jacobian to be differentible
    :param allow_unused: set False to assert all inputs affected the outputs
    :returns: a tensor of size (batchsize, *outsize, *insize) containing the
        jacobian of outputs with respect to inputs for batch element i in row i
    """
    batchsize = outputs.size()[0]
    outsize = outputs.size()[1:]
    insize = inputs.size()[1:]

    jac = outputs.new_zeros((batchsize, *outsize, *insize)
                            ).view(batchsize, -1, *insize)
    flat_out = outputs.reshape(batchsize, -1)
    for i in range(jac.size()[1]):
        col_i = autograd.grad(flat_out[:, i], inputs, grad_outputs=torch.eye(batchsize), retain_graph=True,
                              create_graph=create_graph, allow_unused=allow_unused)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[:, i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(batchsize, *outsize, *insize)


def jacobian(outs, ins, batched=False, create_graph=False, allow_unused=False):
    """Computes the jacobian of outs with respect to in

    :param outs: output of some tensor function
    :param ins: input to some tensor function
    :param batched: whether the first dimension of outs and ins is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :param allow_unused: whether terms in ins are allowed to not contribute to
        any of outs
    :returns jacobian. Size (*outs.size(), *ins.size()) or 
        (batchsize, *outs.size()[1:], *ins.size()[1:])
    """
    if batched:
        return _batched_jacobian(outs, ins, create_graph=create_graph,
                                 allow_unused=allow_unused)
    else:
        return _jacobian(outs, ins, create_graph=create_graph,
                         allow_unused=allow_unused)


def jacobian_and_hessian(outs, ins, batched=False, create_graph=False, allow_unused=False):
    """Computes the jacobian and the hessian of outs with respect to ins

    :param outs: output of some tensor function
    :param ins: input to some tensor function
    :param batched: whether the first dimension of outs and ins is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :param allow_unused: whether terms in ins are allowed to not contribute to
        any of outs
    :returns jacobian, hessian
    """
    jac = jacobian(outs, ins, batched=batched,
                   create_graph=True, allow_unused=allow_unused)
    hes = jacobian(jac, ins, batched=batched, create_graph=create_graph,
                   allow_unused=allow_unused)
    return jac, hes


def trace(tensor):
    """Computes the trace of the last two dimensions of a tensor

    :returns: the trace of the tensor. Size (1,) or (tensor.size()[0], 1)
    """
    return torch.einsum('...ii->...', tensor)


def divergence(outs, ins, jacobian=None, batched=False, create_graph=False, allow_unused=False):
    """Computes the divergence of outs with respect to ins. If jacobian is 
    provided, then will be more efficient

    :param outs: output of some tensor function
    :param ins: input to some tensor function
    :param jacobian: optional parameter for the jacobian of out w.r.t. in
    :param batched: whether the first dimension of outs and ins is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param allow_unused: whether terms in ins are allowed to not contribute to
        any of outs
    :returns: divergence tensor. Size (1,) or (out.size()[0], 1)
    """
    if jacobian is None:
        jacobian = jacobian(
            outs, ins, batched=batched, create_graph=create_graph, allow_unused=allow_unused)
    return trace(jacobian)


def jacobian_and_laplacian(outs, ins, batched=False, create_graph=False, allow_unused=False):
    """This currently computes the laplacian by using the entire hessian. There
    may be a more efficient way to do this

    :param outs: output of some tensor function
    :param ins: input to some tensor function
    :param batched: whether the first dimension of outs and ins is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :returns jacobian, laplacian
    """
    jac, hes = jacobian_and_hessian(
        outs, ins, batched=batched, create_graph=create_graph, allow_unused=allow_unused)
    lap = trace(hes)
    return jac, lap
