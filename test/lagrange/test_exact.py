import numpy as np
import pytest
import torch

from src.lagrange.exact import compute_exact_multipliers


def test_compute_exact_multipliers():

    rand_size = np.random.randint(2, 10)
    batch_size = np.random.randint(2, 10)

    # single constraint
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins[0]

    multipliers = compute_exact_multipliers(loss, constraint, [ins])
    j_g = torch.zeros((1, rand_size))
    j_g[0, 0] = 1.0
    j_fT = torch.ones((rand_size, 1))
    expected = (constraint - j_g @ j_fT).view(constraint.size())
    assert torch.allclose(multipliers, expected)

    # multiple constraints
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins
    multipliers = compute_exact_multipliers(loss, constraint, [ins])
    j_g = torch.eye(rand_size)
    j_fT = torch.ones((rand_size, 1))
    expected = (constraint.view(-1, 1) - j_g @ j_fT).view(constraint.size())
    assert torch.allclose(multipliers, expected)

    # batched single constraints
    ins = torch.rand(batch_size, rand_size, requires_grad=True)
    loss = torch.sum(ins, dim=-1)
    constraint = ins[:, 0].view(-1)

    multipliers = compute_exact_multipliers(loss, constraint, [ins])
    j_g = torch.zeros((batch_size, 1, rand_size))
    j_g[:, 0, 0] = 1.0
    j_fT = torch.ones((batch_size, rand_size, 1))
    expected = (constraint.view(-1, 1, 1) - torch.bmm(j_g, j_fT)).view(
        constraint.size()
    )

    assert torch.allclose(multipliers, expected)

    # batched multiple constraints
    ins = torch.rand(batch_size, rand_size, requires_grad=True)
    loss = torch.sum(ins, dim=-1)
    constraint = ins

    multipliers = compute_exact_multipliers(loss, constraint, [ins])
    j_g = torch.eye(rand_size).expand(batch_size, -1, -1)
    j_fT = torch.ones((batch_size, rand_size, 1))

    expected = (constraint.unsqueeze(-1) - torch.bmm(j_g, j_fT)).view(
        constraint.size()
    )

    assert torch.allclose(multipliers, expected)

    # separate parameters
    other_rand_size = np.random.randint(2, 10)

    ins1 = torch.rand(batch_size, rand_size, requires_grad=True)
    ins2 = torch.rand(
        batch_size, rand_size, other_rand_size, requires_grad=True
    )
    loss = torch.sum(ins1, dim=-1) + torch.sum(ins2, dim=(-1, -2))
    constraint = (ins1[:, 0] + ins2[:, 0, 0]).view(-1)

    multipliers = compute_exact_multipliers(loss, constraint, [ins1, ins2])
    j_g = torch.zeros((batch_size, 1, rand_size + other_rand_size * rand_size))
    j_g[:, 0, 0] = 1.0
    j_g[:, 0, rand_size] = 1.0
    j_fT = torch.ones((batch_size, rand_size + other_rand_size * rand_size, 1))
    expected = (constraint.view(-1, 1, 1) - torch.bmm(j_g, j_fT)).view(
        constraint.size()
    )

    # non-full rank with batching
    ins = torch.rand(batch_size, rand_size, requires_grad=True)
    loss = torch.sum(ins, dim=-1)
    constraint = torch.stack([ins[:, 0], ins[:, 0]], dim=-1)

    with pytest.raises(RuntimeError):
        multipliers = compute_exact_multipliers(
            loss, constraint, [ins], warn="error"
        )
