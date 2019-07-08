import torch
from torch import autograd


__all__ = ["jacobian", "jacobian_and_hessian", "trace", "divergence",
           "jacobian_and_laplacian"]


def _get_size(tensor):
    """Returns the size of a tensor, but treats a scalar as a 1-element vector"""
    if tensor.dim() == 0:
        return torch.Size([1])
    else:
        return tensor.size()


def _jacobian(outputs, inputs, create_graph, allow_unused):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: a list of tensors for the inputs of some function
    :param create_graph: set True for the resulting jacobian to be differentible
    :param allow_unused: set False to assert all inputs affected the outputs
    :returns: a list of tensors of size (outputs.size() + inputs[i].size())
        containing the jacobians of outputs with respect to inputs
    """

    jacs = [outputs.new_zeros((*_get_size(outputs), *_get_size(ins))).view(-1, *_get_size(ins))
            for ins in inputs]
    for i, out in enumerate(outputs.view(-1)):
        cols_i = autograd.grad(out, inputs, retain_graph=True,
                               create_graph=create_graph, allow_unused=allow_unused)

        cols_A = [autograd.grad(out, ins, retain_graph=True,
                                create_graph=create_graph, allow_unused=allow_unused)[0] for ins in inputs]
        cols_B = list(autograd.grad(out, inputs, retain_graph=True,
                                    create_graph=create_graph, allow_unused=allow_unused))

        for j, col_i in enumerate(cols_i):
            if col_i is None:
                # this element of output doesn't depend on the inputs, so leave gradient 0
                continue
            else:
                jacs[j][i] = col_i

    for j in range(len(jacs)):
        if create_graph:
            jacs[j].requires_grad_()
        jacs[j] = jacs[j].view(*_get_size(outputs), *_get_size(inputs[j]))

    return jacs


def _batched_jacobian(outputs, inputs, create_graph, allow_unused):
    """Computes the jacobian of a vector batched outputs with respected to inputs

    :param outputs: tensor for the output of some vector function.
        size: (batchsize, *outsize)
    :param inputs: list of tensors for the inputs of some vector function.
        size: (batchsize, *insize[i]) for each element of inputs
    :param create_graph: set True for the resulting jacobian to be differentible
    :param allow_unused: set False to assert all inputs affected the outputs
    :returns: a list of tensors of size (batchsize, *outsize, *insize[i])
        containing the jacobian of outputs with respect to inputs for batch
        element b in row b
    """
    batchsize = _get_size(outputs)[0]
    outsize = _get_size(outputs)[1:]

    jacs = [outputs.new_zeros(
            (batchsize, *outsize, *_get_size(ins)[1:])
            ).view(batchsize, -1, *_get_size(ins)[1:]) for ins in inputs]
    flat_out = outputs.reshape(batchsize, -1)
    for i in range(flat_out.size()[1]):
        cols_i = autograd.grad(flat_out[:, i], inputs, grad_outputs=torch.eye(batchsize), retain_graph=True,
                               create_graph=create_graph, allow_unused=allow_unused)
        for j, col_i in enumerate(cols_i):
            if col_i is None:
                # this element of output doesn't depend on the inputs, so leave gradient 0
                continue
            else:
                jacs[j][:, i] = col_i

    for j in range(len(jacs)):
        if create_graph:
            jacs[j].requires_grad_()
        jacs[j] = jacs[j].view(batchsize, *outsize, *_get_size(inputs[j])[1:])

    return jacs


def jacobian(outs, ins, batched=False, create_graph=False, allow_unused=False):
    """Computes the jacobian of outs with respect to in

    :param outs: output of some tensor function
    :param ins: either a single tensor input to some tensor function or a list
        of tensor inputs to a function
    :param batched: whether the first dimension of outs and ins is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :param allow_unused: whether terms in ins are allowed to not contribute to
        any of outs
    :returns: jacobian(s) of the same shape as ins. Each jacobian will have 
        size (*outs.size(), *ins[i].size()) or 
        (batchsize, *outs.size()[1:], *ins[i].size()[1:])
    """
    if isinstance(ins, list) or isinstance(ins, tuple):
        if batched:
            return _batched_jacobian(outs, ins, create_graph=create_graph,
                                     allow_unused=allow_unused)
        else:
            return _jacobian(outs, ins, create_graph=create_graph,
                             allow_unused=allow_unused)
    else:
        ins_list = [ins]
        if batched:
            return _batched_jacobian(outs, ins_list, create_graph=create_graph,
                                     allow_unused=allow_unused)[0]
        else:
            return _jacobian(outs, ins_list, create_graph=create_graph,
                             allow_unused=allow_unused)[0]


def jacobian_and_hessian(outs, ins, batched=False, create_graph=False, allow_unused=False):
    """Computes the jacobian and the hessian of outs with respect to ins

    :param outs: output of some tensor function
    :param ins: either a single tensor input to some tensor function or a list
        of tensor inputs to a function
    :param batched: whether the first dimension of outs and ins is actually a
        batch dimension (i.e. the function was applied to an entire batch)
    :param create_graph: whether the resulting hessian should be differentiable
    :param allow_unused: whether terms in ins are allowed to not contribute to
        any of outs
    :returns jacobian, hessian, which are lists if ins is a list
    """
    jac = jacobian(outs, ins, batched=batched,
                   create_graph=True, allow_unused=allow_unused)
    if isinstance(jac, list):
        hes = [jacobian(jac_i, ins,  batched=batched, create_graph=create_graph,
                        allow_unused=allow_unused) for jac_i in jac]
    else:
        hes = jacobian(jac, ins, batched=batched, create_graph=create_graph,
                       allow_unused=allow_unused)
    return jac, hes


def trace(tensor):
    """Computes the trace of the last two dimensions of a tensor

    :param tensor: either a single tensor or a list of tensors
    :returns: either a single tensor of a list of tensors containing the trace 
        of the given tensor(s), with each of size (*tensor[i].size()[:-2], 1)
    """
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return [torch.einsum('...ii->...', t) for t in tensor]
    else:
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
