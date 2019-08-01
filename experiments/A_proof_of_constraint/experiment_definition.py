"""Defines all of the configurations which we want to evaluate"""

import itertools
import sys
from torch import nn


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
            "training_sampling": ["uniform", "start"],
            "num_points": [1000],
            "num_training": [500],
            "batch_size": [100],
            "model_size": [[20, 20, 20]],
            "learning_rate": [1e-3],
            "method": ["average", "batchwise", "unconstrained"],
            "model_act": [nn.Tanh(), nn.SELU()],
            "num_epochs": [200],
            "save_directory": ["results/checkpoints"],
            "save_interval": [10],
        }
    )
)

if __name__ == "__main__":
    # write out the number of configurations to the batch script which called this
    sys.stdout.write(f"{len(CONFIGURATIONS)}")
    sys.stdout.flush()
