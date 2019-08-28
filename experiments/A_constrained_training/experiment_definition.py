"""Defines all of the configurations which we want to evaluate"""

import itertools
import numpy as np
import sys
from torch import nn

from experiments.A_constrained_training.constraints import (
    helmholtz_equation,
    pythagorean_equation,
)

from experiments.A_constrained_training.model import (
    Dense,
    ParameterizedDense,
    Swish,
)
from experiments.A_constrained_training.reductions import (
    Huber_Reduction,
    Lp_Reduction,
)


def get_configuration(index):
    return CONFIGURATIONS[index]


def dictionary_product(**kwargs):
    """Generator which converts a dictionary of lists to a
    list of dictionaries representing the products cartesian products

    e.g. {"number": [1,2,3], "color": ["orange","blue"] } ->
     [{"number": 1, "color": "orange"},
      {"number": 1, "color": "blue"},
      {"number": 2, "color": "orange"},
      {"number": 2, "color": "blue"},
      {"number": 3, "color": "orange"},
      {"number": 3, "color": "blue"}]
    """
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))


CONFIGURATIONS = list(
    dictionary_product(
        **{
            "training_parameterizations": [
                {
                    "amplitudes": np.linspace(0.2, 5.0, num=10),
                    "frequencies": np.linspace(0.2, 5.0, num=10),
                    "phases": np.linspace(-0.5, 0.5, num=10),
                    "num_points": 50,
                    "sampling": "random",
                }
            ],
            "testing_parameterizations": [
                {
                    "amplitudes": [1.0],
                    "frequencies": [1.0],
                    "phases": [0.0],
                    "num_points": 500,
                    "sampling": "uniform",
                }
            ],
            "batch_size": [1000],
            "architecture": [ParameterizedDense],
            "model_size": [[50, 50, 50, 50, 50]],
            "learning_rate": [1e-3],
            "reduction": [Huber_Reduction(6)],
            "model_act": [nn.Tanh(), Swish()],
            "num_epochs": [500],
            "save_directory": ["results/checkpoints"],
            "save_interval": [10],
            "method": [
                "unconstrained",
                "soft-constrained",
                "reduction",
                # "constrained",
            ],
            "constraint": [helmholtz_equation],
        }
    )
)

if __name__ == "__main__":
    # write out the number of configurations to the batch script which called this
    sys.stdout.write(f"{len(CONFIGURATIONS)}")
    sys.stdout.flush()
