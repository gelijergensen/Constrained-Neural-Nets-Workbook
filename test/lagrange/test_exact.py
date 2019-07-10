import numpy as np
import pytest
import torch

from pyinsulate.lagrange.exact import _constrain_loss


def test_constrain_loss():

    rand_size = np.random.randint(2, 10)
    batch_size = np.random.randint(2, 10)

    # single constraint
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins[0]

    con_loss = _constrain_loss(loss, constraint, [ins])
    j_g = torch.zeros((1, rand_size))
    j_g[0, 0] = 1.
    j_fT = torch.ones((rand_size, 1))
    expected = loss + (constraint - j_g @ j_fT).view([]) * constraint
    assert(torch.allclose(con_loss, expected))

    # multiple constraints
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins
    con_loss = _constrain_loss(loss, constraint, [ins])
    j_g = torch.eye(rand_size)
    j_fT = torch.ones((rand_size, 1))
    expected = loss + \
        torch.dot((constraint.view(-1, 1) - j_g @ j_fT).view(-1), constraint)
    assert(torch.allclose(con_loss, expected))

    # batched single constraints
    ins = torch.rand(batch_size, rand_size, requires_grad=True)
    loss = torch.sum(ins, dim=-1)
    constraint = ins[:, 0].view(-1)

    con_loss = _constrain_loss(loss, constraint, [ins])
    j_g = torch.zeros((batch_size, 1, rand_size))
    j_g[:, 0, 0] = 1.
    j_fT = torch.ones((batch_size, rand_size, 1))
    expected = loss + torch.bmm(constraint.view(-1, 1, 1),
                                constraint.view(-1, 1, 1) - torch.bmm(j_g, j_fT)).view(loss.size())

    assert(torch.allclose(con_loss, expected))

    # batched multiple constraints
    ins = torch.rand(batch_size, rand_size, requires_grad=True)
    loss = torch.sum(ins, dim=-1)
    constraint = ins

    con_loss = _constrain_loss(loss, constraint, [ins])
    j_g = torch.eye(rand_size).expand(batch_size, -1, -1)
    j_fT = torch.ones((batch_size, rand_size, 1))

    expected = loss + torch.bmm(constraint.unsqueeze(-2),
                                constraint.unsqueeze(-1) - torch.bmm(j_g, j_fT)).view(loss.size())

    assert(torch.allclose(con_loss, expected))

    # separate parameters
    ins1 = torch.rand(batch_size, rand_size, requires_grad=True)
    ins2 = torch.rand(batch_size, rand_size, requires_grad=True)
    loss = torch.sum(ins1, dim=-1) + torch.sum(ins2, dim=-1)
    constraint = (ins1 + ins2)[:, 0].view(-1)

    con_loss = _constrain_loss(loss, constraint, [ins1, ins2])
    j_g = torch.zeros((batch_size, 1, rand_size*2))
    j_g[:, 0, 0::rand_size] = 1.
    j_fT = torch.ones((batch_size, rand_size*2, 1))
    expected = loss + torch.bmm(constraint.view(-1, 1, 1),
                                constraint.view(-1, 1, 1) - torch.bmm(j_g, j_fT)).view(loss.size())

    # non-full rank
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = torch.stack([ins[0], ins[0]])

    with pytest.raises(RuntimeError):
        con_loss = _constrain_loss(loss, constraint, [ins], warn="error")
