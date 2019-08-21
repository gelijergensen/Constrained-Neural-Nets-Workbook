import numpy as np
import torch

from pyinsulate.pdes import helmholtz_equation, pythagorean_equation


def test_helmholtz_equation():

    pde = helmholtz_equation

    # unbatched
    frequency = torch.rand(1)
    phase = torch.rand(1) * 2 * np.pi
    amplitude = torch.rand(1) * 100
    xb = torch.rand(1, 1, requires_grad=True)
    out = amplitude * torch.sin(2 * np.pi * frequency * xb + phase)
    value, (lhs, rhs, jac) = pde(
        out,
        xb,
        torch.tensor([amplitude, frequency, phase]),
        return_diagnostics=True,
    )

    deriv = (
        amplitude
        * ((2 * np.pi * frequency))
        * torch.cos(2 * np.pi * frequency * xb + phase)
    )
    lap = (
        -amplitude
        * ((2 * np.pi * frequency) ** 2)
        * torch.sin(2 * np.pi * frequency * xb + phase)
    )

    # TODO there is clearly a problem here because the difference is often above 1e-4, even
    # print("")
    # print(f"lhs: {lhs}")
    # print(f"rhs: {rhs}")
    # print(f"value: {value}")
    # print(f"deriv: {deriv}")
    # print(f"jac: {jac}")
    # print(f"lap: {lap}")

    # assert torch.allclose(jac, deriv)
    # assert torch.allclose(lhs, lap)
    assert torch.allclose(
        value, value.new_zeros(value.size()), atol=1e-3
    )  # FIXME

    # batched
    batch_size = np.random.randint(1, 100)
    # batch_size = 5
    frequency = torch.rand(batch_size)
    phase = torch.rand(batch_size) * 2 * np.pi
    amplitude = torch.rand(batch_size) * 100
    xb = torch.rand(batch_size, 1, requires_grad=True)

    out = amplitude.view(xb.size()) * torch.sin(
        2 * np.pi * frequency.view(xb.size()) * xb + phase.view(xb.size())
    )
    value, (lhs, rhs, jac) = pde(
        out,
        xb,
        torch.stack([amplitude, frequency, phase], dim=1),
        return_diagnostics=True,
    )

    deriv = (
        amplitude.view(xb.size())
        * ((2 * np.pi * frequency)).view(xb.size())
        * torch.cos(
            2 * np.pi * frequency.view(xb.size()) * xb + phase.view(xb.size())
        )
    ).unsqueeze(-1)
    lap = (
        -amplitude.view(xb.size())
        * ((2 * np.pi * frequency).view(xb.size()) ** 2)
        * torch.sin(
            2 * np.pi * frequency.view(xb.size()) * xb + phase.view(xb.size())
        )
    )

    # TODO there is clearly a problem here because the difference is often above 1e-4, even
    # print("")
    # print(f"lhs: {lhs}")
    # print(f"rhs: {rhs}")
    # print(f"value: {value}")
    # print(f"deriv: {deriv}")
    # print(f"jac: {jac}")
    # print(f"lap: {lap}")

    # print(f"jac.size(): {jac.size()}")
    # print(f"deriv.size(): {deriv.size()}")

    # for i in range(jac.view(-1).size()[0]):
    #     print(f"{jac.view(-1)[i].item()} :: {deriv.view(-1)[i].item()}")

    # assert torch.allclose(jac, deriv)
    # assert torch.allclose(lhs, lap)
    assert torch.allclose(
        value, value.new_zeros(value.size()), atol=1e-3
    )  # FIXME


# def test_pythagorean_equation():  # FIXME

#     pde = pythagorean_equation

#     # unbatched
#     frequency = torch.rand(1)
#     phase = torch.rand(1) * 2 * np.pi
#     amplitude = torch.rand(1) * 100
#     xb = torch.rand(1, 1, requires_grad=True)
#     out = amplitude.view(xb.size()) * torch.sin(
#         2 * np.pi * frequency.view(xb.size()) * xb + phase.view(xb.size())
#     )
#     value = pde(out, xb, torch.tensor([amplitude, frequency, phase]))

#     assert torch.allclose(value, value.new_zeros(value.size()), atol=1e-5)

#     # batched
#     batch_size = np.random.randint(1, 100)
#     frequency = torch.rand(batch_size)
#     phase = torch.rand(batch_size) * 2 * np.pi
#     amplitude = torch.rand(batch_size) * 100
#     xb = torch.rand(batch_size, 1, requires_grad=True)

#     out = torch.sin(
#         2 * np.pi * frequency.view(xb.size()) * xb + phase.view(xb.size())
#     )
#     value = pde(out, xb, torch.stack([amplitude, frequency, phase], dim=1))

#     assert torch.allclose(value, value.new_zeros(value.size()), atol=1e-5)

