from ignite.engine import Engine
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ignite import GradientConstraint


def test_GradientConstraint():

    # A fake constraint which assumes two given inputs
    def constraint(x, y, **kwargs):
        if len(kwargs) == 0:
            print("This happened")
            return torch.tensor([0])
        else:
            return x

    metrics = {
        "dummy_constraint": GradientConstraint(
            constraint,
            output_transform=lambda args: (
                args[0],
                args[0],
                {"dummy_kwarg": "dummy_value"},
            ),
        ),
        "dummy_sum": GradientConstraint(
            constraint,
            output_transform=lambda args: (
                args[0],
                args[0],
                {"dummy_kwarg": "dummy_value"},
            ),
        )
        + GradientConstraint(
            constraint,
            output_transform=lambda args: (
                args[0],
                args[0],
                {"dummy_kwarg": "dummy_value"},
            ),
        ),
    }

    # A dummy update loop
    def _update(engine, batch):
        return batch

    engine = Engine(_update)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    # Dummy data
    dummy_dl = DataLoader(
        TensorDataset(torch.tensor([[1, 1, 1]])), batch_size=1
    )

    engine.run(dummy_dl, 2)

    print(
        f'engine.state.metrics["dummy_constraint"]: {engine.state.metrics["dummy_constraint"]}'
    )

    # Check the kwargs were passed through
    assert len(engine.state.metrics["dummy_constraint"].size()) > 1
    # Check the constraint summing worked
    assert torch.allclose(
        engine.state.metrics["dummy_sum"],
        2 * engine.state.metrics["dummy_constraint"],
    )
