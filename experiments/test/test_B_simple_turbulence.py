import os

import torch.nn as nn

from ..B_simple_turbulence.simple_turbulence import run_analysis, load_data


def test_simple_turbulence():
    path = os.path.expandvars("$SCRATCH/data/divfree-test/raw_0100.npy")
    train_dl, test_dl = load_data(
        path, 32 * 32 * 32, num_testing=128, batch_size=128
    )

    final_loss = run_analysis(
        train_dl, test_dl, [20], nn.LeakyReLU(), max_epochs=1
    )
    assert True  # if it runs, we are good
