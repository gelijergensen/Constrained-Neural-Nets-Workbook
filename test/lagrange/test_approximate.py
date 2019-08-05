import numpy as np
import pytest
import torch

from pyinsulate.lagrange.exact import compute_exact_multipliers
from pyinsulate.lagrange.approximate import compute_approximate_multipliers


def test_compute_approximate_multipliers():

    rand_size = np.random.randint(2, 10)
    batch_size = np.random.randint(2, 10)

    # Single constraint without state
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins[0]
    expected_multipliers = compute_exact_multipliers(loss, constraint, [ins])
    multipliers, state = compute_approximate_multipliers(
        loss, constraint, [ins], state=None
    )
    assert torch.allclose(multipliers, expected_multipliers)
    # Single constraint with state
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins[0]
    expected_multipliers = compute_exact_multipliers(loss, constraint, [ins])
    multipliers, state = compute_approximate_multipliers(
        loss, constraint, [ins], state=state
    )
    assert torch.allclose(multipliers, expected_multipliers, atol=1e-3)

    # Multiple constraints without state
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins
    expected_multipliers = compute_exact_multipliers(loss, constraint, [ins])
    multipliers, state = compute_approximate_multipliers(
        loss, constraint, [ins], state=None
    )
    assert torch.allclose(multipliers, expected_multipliers)
    # Multiple constraints with state
    ins = torch.rand(rand_size, requires_grad=True)
    loss = torch.sum(ins)
    constraint = ins
    expected_multipliers = compute_exact_multipliers(loss, constraint, [ins])
    multipliers, state = compute_approximate_multipliers(
        loss, constraint, [ins], state=state
    )
    assert torch.allclose(multipliers, expected_multipliers, atol=1e-3)

    # Separate parameters without state
    other_rand_size = np.random.randint(2, 10)

    ins1 = torch.rand(rand_size, requires_grad=True)
    ins2 = torch.rand(rand_size, other_rand_size, requires_grad=True)
    loss = torch.sum(ins1, dim=-1) + torch.sum(ins2, dim=(-1, -2))
    constraint = (ins1[0] + ins2[0, 0]).view(-1)
    expected_multipliers = compute_exact_multipliers(
        loss, constraint, [ins1, ins2]
    )
    multipliers, state = compute_approximate_multipliers(
        loss, constraint, [ins1, ins2], state=None
    )
    assert torch.allclose(multipliers, expected_multipliers)
    # Separate parameters with state
    ins1 = torch.rand(rand_size, requires_grad=True)
    ins2 = torch.rand(rand_size, other_rand_size, requires_grad=True)
    loss = torch.sum(ins1, dim=-1) + torch.sum(ins2, dim=(-1, -2))
    constraint = (ins1[0] + ins2[0, 0]).view(-1)
    expected_multipliers = compute_exact_multipliers(
        loss, constraint, [ins1, ins2]
    )
    multipliers, state = compute_approximate_multipliers(
        loss, constraint, [ins1, ins2], state=state
    )
    assert torch.allclose(multipliers, expected_multipliers, atol=1e-3)
