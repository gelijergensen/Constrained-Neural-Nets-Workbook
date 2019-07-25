"""Defines all of the configurations which we want to evaluate"""

import itertools
import sys


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
            "training_sampling": ["uniform"],
            "num_points": [100],
            "num_training": [10],
            "batch_size": [10],
            "model_size": [[50], [25, 25]],
            "learning_rate": [1e-2],
        }
    )
)

if __name__ == "__main__":
    # write out the number of configurations to the batch script which called this
    sys.stdout.write(f"{len(CONFIGURATIONS)}")
    sys.stdout.flush()
