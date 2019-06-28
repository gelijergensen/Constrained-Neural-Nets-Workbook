import torch
from torch import autograd


def jacobian(outputs, inputs, create_graph=False, allow_unused=True):
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


def batched_jacobian(outputs, inputs, create_graph=False, allow_unused=True):
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
