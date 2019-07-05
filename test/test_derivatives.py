import numpy as np
import torch

from pyinsulate.derivatives import jacobian, trace


def test_jacobian():

    batchsize = int(np.random.randint(1, 10))

    # vector * matrix  --

    # Unbatched
    rand_lengths = np.random.randint(1, 10, 2)
    ins = torch.rand(tuple(list(rand_lengths[-1:])), requires_grad=True)
    factor = torch.rand(tuple(list(rand_lengths)))
    out = factor @ ins
    jac = jacobian(out, ins)
    assert(torch.allclose(jac, factor))

    # Batched
    ins = ins.unsqueeze(0).expand(batchsize, *ins.size())
    out = torch.einsum('ij,kj->ki', factor, ins)
    assert(torch.allclose(torch.squeeze(out), out))
    bat_jac = jacobian(out, ins, batched=True)
    for i in range(batchsize):
        assert(torch.allclose(bat_jac[i], factor))

    # matrix * matrix  --

    # Unbatched
    rand_lengths = np.random.randint(1, 10, 3)
    ins = torch.rand(tuple(list(rand_lengths[-2:])), requires_grad=True)
    factor = torch.rand(tuple(list(rand_lengths[:-1])))

    out = factor @ ins
    jac = jacobian(out, ins)
    ans = jac.new_zeros(jac.size())
    for i in range(jac.size()[-1]):
        ans[:, i, :, i] = factor
    assert(torch.allclose(jac, ans))

    # Batched
    ins = ins.unsqueeze(0).expand(batchsize, *ins.size())
    out = torch.einsum('ij,kjl->kil', factor, ins)
    bat_jac = jacobian(out, ins, batched=True)
    ans = jac.new_zeros(bat_jac.size())
    for b in range(batchsize):
        for i in range(bat_jac.size()[-1]):
            ans[b, :, i, :, i] = factor
    assert(torch.allclose(bat_jac, ans))

    # Confirm agreement in complex case  --

    # Unbatched
    rand_lengths = np.random.randint(1, 10, 5)
    ins = torch.rand(tuple(list(rand_lengths)), requires_grad=True)

    out = torch.relu(ins)
    jac = jacobian(out, ins)

    # Check that lists work correctly
    out = torch.relu(ins)
    list_jac = jacobian(out, [ins, ins])
    assert(all(torch.allclose(jac, list_jac[i]) for i in range(len(list_jac))))

    # Batched
    ins = ins.view(-1, *ins.size())

    out = torch.relu(ins)
    bat_jac = jacobian(out, ins, batched=True)

    assert(torch.allclose(jac, bat_jac[0]))


def test_trace():

    # Unbatched
    rand_length = int(np.random.randint(1, 10, 1))
    ins = torch.rand((rand_length, rand_length))
    trc = trace(ins)
    assert(torch.allclose(trc, torch.trace(ins)))

    # Check that lists work correctly
    list_trc = trace([ins, ins])
    assert(all(torch.allclose(list_trc[i], trc)
               for i in range(len(list_trc))))

    # Batched
    batchsize = int(np.random.randint(1, 10))
    ins = ins.unsqueeze(0).expand(batchsize, *ins.size())

    ans = trace(ins)
    for b in range(batchsize):
        assert(torch.allclose(ans[b], trc))
