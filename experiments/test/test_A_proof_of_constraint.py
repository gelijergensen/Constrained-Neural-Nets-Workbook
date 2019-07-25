import glob
import os

import torch
import torch.nn as nn

from ..A_proof_of_constraint.main import run_experiment


def test_proof_of_constraint():

    CHECKPOINT_DIR = ".temp"
    directory = os.path.join(CHECKPOINT_DIR, "test_proof_of_constraint")
    save_file = "quick-test"

    # Delete any files that somehow were left over in this directory
    files = glob.glob(f"{directory}/{save_file}*.pth")
    for f in files:
        os.remove(f)

    num_epochs = 1
    final_result = run_experiment(
        num_epochs,
        save_directory=directory,
        save_file=save_file,
        num_points=100,
        num_training=10,
        batch_size=10,
    )

    # Try to load in the model again
    files = glob.glob(f"{directory}/{save_file}*.pth")

    try:
        assert len(files) == num_epochs

        loaded_result = torch.load(files[-1])

        loaded_config = loaded_result["configuration"]
        final_config = final_result[0]

        # Can't compare the functions directly
        assert type(loaded_config.pop("model_act")) == type(
            final_config.pop("model_act")
        )
        assert loaded_config == final_config  # Remainder compared directly

    except AssertionError as assertFailed:
        failure = assertFailed
    else:
        failure = None

    # cleanup
    for f in files:
        os.remove(f)

    # TODO abstract the model/optimizer construction in main!

    if failure is not None:
        raise failure
