"""Defines all of the configurations which we want to evaluate"""

import itertools
import numpy as np
import sys
from torch import nn

from experiments.B_nonlinear_projection.constraints import (
    helmholtz_equation,
    pythagorean_equation,
)

from experiments.B_nonlinear_projection.model import (
    Dense,
    ParameterizedDense,
    Swish,
)
from experiments.B_nonlinear_projection.error_functions import (
    Huber_Error,
    Lp_Error,
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
            "model_act": [Swish()],
            "num_epochs": [500],
            "save_directory": ["results/checkpoints"],
            "save_interval": [10],
            "device": ["cpu"],
            "constraint": [helmholtz_equation],
            "error_fn": [None, Huber_Error],
            "regularization_weight": np.linspace(0, 1, num=10),
            "tolerance": [1e-5],
            "max_iterations": [1e4],
        }
    )
)

if __name__ == "__main__":
    # write out the number of configurations to the batch script which called this
    sys.stdout.write(f"{len(CONFIGURATIONS)}")
    sys.stdout.flush()
