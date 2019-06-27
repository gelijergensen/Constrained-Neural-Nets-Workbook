import numpy as np
import torch

from pyinsulate.derivatives import jacobian


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
    print(rand_lengths)
    ins = torch.rand(tuple(list(rand_lengths[-2:])), requires_grad=True)
    factor = torch.rand(tuple(list(rand_lengths[:-1])))

    print(f"{factor.size()} @ {ins.size()}")

    out = factor @ ins

    jac = jacobian(out, ins)
    print(jac.size())

    ans = jac.new_zeros(jac.size())
    for i in range(jac.size()[-1]):
        ans[:, i, :, i] = factor
    assert(torch.allclose(jac, ans))
