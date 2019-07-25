import torch

from pyinsulate.losses.pdes import steady_state_turbulence


def test_steady_state_turbulence():

    pde_loss = steady_state_turbulence

    xb = torch.rand(3, requires_grad=True)
    out = torch.relu(xb)
    single_loss = pde_loss(out, xb)

    xb = xb.view(1, -1)
    out = torch.relu(xb)
    batch_loss = pde_loss(out, xb)

    xb = xb.repeat(2, 1)
    out = torch.relu(xb)
    very_batch_loss = pde_loss(out, xb)

    assert torch.allclose(single_loss, batch_loss)
    assert torch.allclose(batch_loss, very_batch_loss)
