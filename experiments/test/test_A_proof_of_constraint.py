import os

import torch.nn as nn

from ..A_proof_of_constraint.proof_of_constraint import run_experiment


def test_proof_of_constraint():
    final_result = run_experiment(
        1, num_points=100, num_training=10, batch_size=10)
    assert(True)  # if it runs, we are good
