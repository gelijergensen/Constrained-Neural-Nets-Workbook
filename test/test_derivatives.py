import numpy as np
import torch

from pyinsulate.derivatives import jacobian, batched_jacobian


def test_jacobian():

    # vector * matrix
    rand_lengths = np.random.randint(1, 10, 2)
    ins = torch.rand(tuple(list(rand_lengths[-1:])), requires_grad=True)
    factor = torch.rand(tuple(list(rand_lengths)))
    out = factor @ ins
    jac = jacobian(out, ins)
    assert(torch.allclose(jac, factor))

    # matrix * matrix
    rand_lengths = np.random.randint(1, 10, 3)
    ins = torch.rand(tuple(list(rand_lengths[-2:])), requires_grad=True)
    factor = torch.rand(tuple(list(rand_lengths[:-1])))

    out = factor @ ins
    jac = jacobian(out, ins)
    ans = jac.new_zeros(jac.size())
    for i in range(jac.size()[-1]):
        ans[:, i, :, i] = factor
    assert(torch.allclose(jac, ans))


def test_batched_jacobian():

    batchsize = 5
    rand_lengths = np.random.randint(1, 10, 2)

    ins = torch.rand((batchsize, rand_lengths[-1]), requires_grad=True)
    out = torch.relu(ins)
    jac = batched_jacobian(out, ins)
    assert(jac.size()[0] == batchsize)
    assert(jac.size()[1] == jac.size()[2] == rand_lengths[-1])

    factor = torch.rand(tuple(list(rand_lengths)))
    out = torch.einsum('ij,kj->ki', factor, ins)
    jac = batched_jacobian(out, ins)
    for i in range(batchsize):
        assert(torch.allclose(jac[i], factor))

    # matrix * matrix
    rand_lengths = np.random.randint(1, 10, 3)
    ins = torch.rand((5,) + tuple(list(rand_lengths[-2:])), requires_grad=True)
    factor = torch.rand(tuple(list(rand_lengths[:-1])))

    out = torch.einsum('ij,kjl->kil', factor, ins)
    for i in range(batchsize):
        assert(torch.allclose(out[i], factor @ ins[i]))
    jac = batched_jacobian(out, ins)
    ans = jac.new_zeros(jac.size())
    for b in range(batchsize):
        for i in range(jac.size()[-1]):
            ans[b, :, i, :, i] = factor
    assert(torch.allclose(jac, ans))
